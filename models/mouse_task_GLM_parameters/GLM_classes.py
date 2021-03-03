import sys
sys.path.insert(1,'/Users/laurence/Desktop/Neuroscience/kevin_projects/code/mousetask/models/two_process_mixture_model')
from instantiate import create_synthetic_data_frame
import glob
from GLM_construct_regressors import load_data
import numpy as np

class Data:
    def __init__(self, data_type):

        if data_type == "s_data":
            print("Synthetic Data Generated")
            self._data_frame = create_synthetic_data_frame()
            self._len_nChoices = len(self._data_frame)

        elif data_type == "b_data":
            print("Behavioural Data Generated")

            # """Pull out all the session files contained within the data folder"""
            sessions = glob.glob("/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/Single_probes_for_cog_model/*.mat")
            data = load_data(sessions[0])
            for session in range(len(sessions) - 1):
                data = data.append(load_data(sessions[session + 1]), ignore_index = True)

            # #test for running one session with kevin-----
            # session = "/Users/laurence/Desktop/Neuroscience/kevin_projects/data/processed_physdata/Single_probes_for_cog_model/aligned_physdata_KM011_2020-03-19_probe0.mat"
            # data = load_data(session)
            #---------------------

            self._data_frame = data
            self._len_nChoices = len(self._data_frame)

            """Choices / Code to convert choices into binary to solve concat error"""
            """Only analyze trials that are non-vilations and of free choice."""
            good_trials = data.loc[(data["free"] == 1)]
            choices = np.asarray(good_trials["left_choices"])
            choices = np.array(choices, dtype=np.float)

            self._nChoices = choices
            self._good_trials = good_trials
