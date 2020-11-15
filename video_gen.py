import numpy as np
import cv2
import matplotlib.pyplot as plt
import PredictLicking.is_licking as lick

# #New trial data 100,101,102
# csv = "/Users/laurence/Desktop/Trial_data_DLC/video_snippet_KM011_2020-03-19_trial102DLC_resnet50_Master_ProjectAug13shuffle1_200000.csv"
# video = "/Users/laurence/Desktop/Trial_data_DLC/video_snippet_KM011_2020-03-19_trial102.avi"

#Test snippet of 24th session - video_snippet
csv = "/Users/laurence/Desktop/Neuroscience/mproject/data/24_3minsDLC_resnet50_Master_ProjectAug13shuffle1_133500.csv"
video = "/Users/laurence/Desktop/file.avi"

#Calculate the frames the mouse is licking
df, array_of_licks = lick.is_licking(csv)
cherry_frames, grape_frames, centre_frames = lick.is_licking_spout(df, csv)

print("###########################")
print("TRIAL snip of 24th")
print("")
print("Frames Licking Cherry", cherry_frames)
print("")
print("Frames Licking Grape", grape_frames)
print("")
print("Frames Licking Centre", centre_frames)
print("")
print("Frames is licking",array_of_licks)
print("###########################")

#Create a videocapture object to ingest the video - Insert file
cap = cv2.VideoCapture(video)

#Setting frame counter to -1 so it aligns with the frame number of each video
frame_counter = -1

#Save video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('CV_generated_video.avi',fourcc,30.0,(640,480),isColor=True)

while(cap.isOpened()):

    # Capture frame-by-frame, ret is true or false and frame captures the frame
    ret, frame = cap.read()
    frame_counter = 1 + frame_counter

    #Place FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                "Frame #:" + str(frame_counter),
                (50, 50),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)

    # Our operations on the frame come here
    colour = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Bug - for some reason the pixel colour only alligns to the correct spout if I switch frame type
    #Cherry Logic - even though I use grape frames
    for x in grape_frames:
        if x == frame_counter:
            #Cherry colour - Y coord vs X coord
            colour[440:450,350:360] = (222, 49, 99)

    #Grape logic - even though I use cherry frames
    for x in cherry_frames:
        if x == frame_counter:
            #Grape colour - Y vs X coord
            colour[440:450,395:405] = (111, 45, 168)

    #centre lick logic
    for x in centre_frames:
        if x == frame_counter:
            colour[440:450,375:385] = (255, 255, 255)

    #output the frame
    out.write(colour)

    # Display the resulting frame
    # Change waitkey to 0 to control speed with a button press
    # cv2.imshow('frame',colour)
    cv2.imshow("video",colour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()
