#This script takes in the CSV output from deep lab cut and processes a data frame with the output required for my project
#This script will not work out of the box for other projects if the DLC labels change

import pandas as pd
import numpy as np

def process_data(file):
    #Read and create df
    read_data = pd.read_csv(file, header=None)
    df = pd.DataFrame(read_data)
    #Change headers
    new_header = df.iloc[2] #Makes x,y and L column headers
    df.columns = new_header #Makes x,y and L column headers
    df = df.drop([0]) #Drops DLC header
    df = df.drop([2]) #Drops 2nd probability header
    #Change data types-----------------------------------------------------
    df.iloc[1:,:] = df.iloc[1:,:].astype(float)
    #Create new df with just features, prob and frames---------------------
    likelihood = df["likelihood"]
    coords = df["coords"] #This column is actually frames videos
    df = likelihood.join(coords)
    df = df.iloc[:,-5:]
    df.columns = ["L_Tongue", "R_Tongue","C_Tongue","FO_Tongue","Frames"]
    df = df.drop([1])
    return(df)
