#This script is uses to decide whether the mouse is licking or not, and which spout
import matplotlib.pyplot as plt
from process_tongue_data import *

#Change file name to run function
df = process_data("/Users/laurence/Desktop/Neuroscience/mproject/Data/402.csv")

#Filter----------------------------------------------------------------
df = df.loc[(df['C_Tongue'] >= 0.99) & (df['FO_Tongue'] >= 0.99)
           |(df['C_Tongue'] >= 0.99) & (df['R_Tongue'] >= 0.99)
           |(df['C_Tongue'] >= 0.99) & (df['L_Tongue'] >= 0.99)
           |(df['L_Tongue'] >= 0.99) & (df['R_Tongue'] >= 0.99)
           |(df['L_Tongue'] >= 0.99) & (df['FO_Tongue'] >= 0.99)
           |(df['R_Tongue'] >= 0.99) & (df['FO_Tongue'] >= 0.99)]

#Plot probability of tongue licking
# df.plot(x='Frames',y= ['L_Tongue','R_Tongue','FO_Tongue','C_Tongue'])
# plt.ylabel("Proability")
# plt.show()

#Create a function to see if the mouse is licking
frame_is_licking = df.iloc[:,-1].values
print(is_licking)
