#This script is uses to decide whether the mouse is licking or not, and which spout
import process_tongue_data as p
from process_tongue_data import *

#Change file name to run function
df = process_data("/Users/laurence/Desktop/Neuroscience/mproject/Data/402.csv")

#Filter----------------------------------------------------------------
def is_licking(processed_data_frame):
    df = df.loc[(df['C_Tongue'] >= 0.99) & (df['FO_Tongue'] >= 0.99)]
print(df)
