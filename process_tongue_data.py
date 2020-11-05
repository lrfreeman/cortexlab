#This script takes in the CSV output from deep lab cut and processes a data frame with the output required for my project
#This script will not work out of the box for other projects if the DLC labels change

import pandas as pd
import numpy as np

# test_file = "/Users/laurence/Desktop/Neuroscience/mproject/data/402.csv"

def process_data_spout(file):
    #Read and create df
    read_data = pd.read_csv(file, header=None, low_memory=False)
    df = pd.DataFrame(read_data)
    #Change headers
    new_header = df.iloc[2] #Makes x,y and L column headers
    df.columns = new_header #Makes x,y and L column headers
    df = df.drop([0]) #Drops DLC header
    df = df.drop([2]) #Drops 2nd probability header
    #Change data types-----------------------------------------------------
    df.iloc[1:,:] = df.iloc[1:,:].astype(float)
    #Rename DF Columns-----------------------------------------------------
    df.columns = ["Frames","LL_GS_X","LL_GS_Y","LL_GS_L",
                           "LC_GS_X","LC_GS_Y","LC_GS_L",
                           "LR_GS_X","LR_GS_Y","LR_GS_L",
                           "RL_CS_X","RL_CS_Y","RL_CS_L",
                           "RC_CS_X","RC_CS_Y","RC_CS_L",
                           "RR_CS_X","RR_CS_Y","RR_CS_L",
                           "LE_T_X", "LE_T_Y", "LE_T_L",
                           "RE_T_X", "RE_T_Y", "RE_T_L",
                           "C_T_X", "C_T_Y", "C_T_L",
                           "F_E_X", "F_E_Y", "F_E_L"]
    df = df.drop([1])
    return(df)

# process_data_spout(test_file)
