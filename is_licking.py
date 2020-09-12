#Run this script with one argument in the command line. That argument should be the csv file output of DLC
#This script is uses to decide whether the mouse is licking or not, and which spout
import matplotlib.pyplot as plt
from process_tongue_data import *
import numpy as np
import cv2
import sys

def is_licking(video_path):
    #Change file name to run function
    df = process_data(video_path)
    #Filter----------------------------------------------------------------
    df = df.loc[(df['C_Tongue'] >= 0.99) & (df['FO_Tongue'] >= 0.99)
               |(df['C_Tongue'] >= 0.99) & (df['R_Tongue'] >= 0.99)
               |(df['C_Tongue'] >= 0.99) & (df['L_Tongue'] >= 0.99)
               |(df['L_Tongue'] >= 0.99) & (df['R_Tongue'] >= 0.99)
               |(df['L_Tongue'] >= 0.99) & (df['FO_Tongue'] >= 0.99)
               |(df['R_Tongue'] >= 0.99) & (df['FO_Tongue'] >= 0.99)]

    #Plot probability of tongue licking------------------------------------
    # df.plot(x='Frames',y= ['L_Tongue','R_Tongue','FO_Tongue','C_Tongue'])
    # plt.ylabel("Proability")
    # plt.show()

    #What frames is the mouse is licking?
    frame_is_licking = df.iloc[:,-1].values
    print()
    print("The mouse is licking at these specifc frames:")
    print()
    print(frame_is_licking)
    print()
    # print(frame_is_licking)

is_licking(sys.argv[1])

#Code to overlay frmes for CML
#ffmpeg -i video.mov -vf "drawtext=fontfile=Arial.ttf: text=%{n}: x=(w-tw)/2: y=h-(2*lh): fontcolor=white: box=1: boxcolor=0x00000099" -y output.mov
