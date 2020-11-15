#This script takes in the CSV output from deep lab cut and processes a data frame with the output required for my project
import pandas as pd
import numpy as np

#Test csv
# csv = "/Users/laurence/Desktop/video_snippet_KM011_2020-03-19_trial98DLC_resnet50_Master_ProjectAug13shuffle1_42000.csv"
# pd.set_option("display.max_rows", None, "display.max_columns", None)

def process_data_spout(file):

    #Read and create df
    read_data = pd.read_csv(file, header=None, float_precision="round_trip", dtype="a")
    df = pd.DataFrame(read_data)

    #Change headers
    new_header = df.iloc[2] #Makes x,y and L column headers
    df.columns = new_header #Makes x,y and L column headers
    df = df.drop([0]) #Drops DLC header
    df = df.drop([2]) #Drops 2nd probability header

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
                           "F_E_X", "F_E_Y", "F_E_L",
                           "L_Hand_X", "L_Hand_Y", "L_Hand_prob",
                           "L1_Hand_X", "L1_Hand_Y", "L1_Hand_prob",
                           "L2_Hand_X", "L2_Hand_Y", "L2_Hand_prob",
                           "L3_Hand_X", "L3_Hand_Y", "L3_Hand_prob",
                           "L4_Hand_X", "L4_Hand_Y", "L4_Hand_prob",
                           "R_Hand_X", "R_Hand_Y", "R_Hand_prob",
                           "R1_Hand_X", "R1_Hand_Y", "R1_Hand_prob",
                           "R2_Hand_X", "R2_Hand_Y", "R2_Hand_prob",
                           "R3_Hand_X", "R3_Hand_Y", "R3_Hand_prob",
                           "R4_Hand_X", "R4_Hand_Y", "R4_Hand_prob"]

    #Drop a row that is string: bodyparts, ll_grape_spout etc..
    df = df.drop([1])
    return(df)

#Test
# df = process_data_spout(csv)
