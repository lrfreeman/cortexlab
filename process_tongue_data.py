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
    df = df.drop([2]) #Drops 2nd probaiblity header
    #Change data types-----------------------------------------------------
    df.iloc[1:,:] = df.iloc[1:,:].astype(float)
    #Create new df with just features, prob and frames---------------------
    likelihood = df["likelihood"]
    coords = df["coords"]
    df = likelihood.join(coords)
    df = df.iloc[:,-5:]
    df.columns = ["L_Tongue", "R_Tongue","C_Tongue","FO_Tongue","Frames"]
    df = df.drop([1])

# #Change file name to run function
# is_licking("/Users/laurence/Desktop/Neuroscience/mproject/Data/402.csv")

# #Read and create df----------------------------------------------------
# read_data = pd.read_csv("/Users/laurence/Desktop/Neuroscience/mproject/Data/402.csv", header=None)
# df = pd.DataFrame(read_data)
# #print(df)
#
# #Change headers--------------------------------------------------------
# new_header = df.iloc[2] #Makes x,y and L column headers
# df.columns = new_header #Makes x,y and L column headers
# df = df.drop([0]) #Drops DLC header
# df = df.drop([2]) #Drops 2nd probaiblity header
# #print(df)
#
# #Change data types-----------------------------------------------------
# df.iloc[1:,:] = df.iloc[1:,:].astype(float)
# # df.iloc[1:,:] = df.iloc[1:,:].apply(pd.to_numeric)
# # print(df.dtypes)
#
# #Create new df with just features, prob and frames---------------------
# likelihood = df["likelihood"]
# coords = df["coords"]
# df = likelihood.join(coords)
# df = df.iloc[:,-5:]
# df.columns = ["L_Tongue", "R_Tongue","C_Tongue","FO_Tongue","Frames"]
# df = df.drop([1])
# # print(df)
#
# #Filter----------------------------------------------------------------
# is_licking = df.loc[(df['C_Tongue'] >= 0.99) & (df['FO_Tongue'] >= 0.99)]
# # df.loc[(df['L_Tongue'] >= 0.99) & (df['R_Tongue'] <= B)]
# # print(df[(df > 0.95).any(1)])
# print(is_licking)
