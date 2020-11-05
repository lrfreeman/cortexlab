import numpy as np
import cv2
import matplotlib.pyplot as plt
from is_licking import *

csv = "/Users/laurence/Desktop/Neuroscience/mproject/data/Analysed_videos_CSVs_DLC/video_snippet_KM011_2020-03-19_trial93DLC_resnet50_Master_ProjectAug13shuffle1_200000.csv"
video = '/Users/laurence/Desktop/Mvideos/Snipp1_500/video_snippet_KM011_2020-03-19_trial93.avi'

#Code to overlay frmes for CML
#ffmpeg -i video.mov -vf "drawtext=fontfile=Arial.ttf: text=%{n}: x=(w-tw)/2: y=h-(2*lh): fontcolor=white: box=1: boxcolor=0x00000099" -y output.mov

#Calculate the frames the mouse is licking
df, frames_licking = is_licking(csv)
cherry_frames, grape_frames, centre_frames = is_licking_spout(df, csv)

#Create a videocapture object to ingest the video - Insert file
cap = cv2.VideoCapture(video)

#Setting frame counter to -1 so it aligns with the frame number of each video which is 235
#bug warning - it should be -2 to match frame numbers, but -1 looks better....
frame_counter = -1

#Save video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('trial93DLC.avi',fourcc,7.0,(640,480),isColor=True)

while(cap.isOpened()):
    # Capture frame-by-frame, ret is true or false and frame captures the frame
    ret, frame = cap.read()
    frame_counter = 1 + frame_counter
    # print(frame_counter)

    # Our operations on the frame come here
    colour = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Grape logic
    for x in grape_frames:
        if x == frame_counter:
            colour[440:450,350:360] = (111, 45, 168)

    #Cherry Logic
    for x in cherry_frames:
        if x == frame_counter:
            colour[440:450,395:405] = (222, 49, 99)

    #centre lick logic
    for x in centre_frames:
        if x == frame_counter:
            colour[440:450,375:385] = (255, 255, 255)

    #output the frame
    out.write(colour)

    # Display the resulting frame
    # Change waitkey to 0 to control speed with a button press
    cv2.imshow('frame',colour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
