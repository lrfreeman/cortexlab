from scipy.io import loadmat
import pandas as pd

file = "/Users/laurence/Desktop/Neuroscience/kevin_projects/data/mousedatas_with_reward_probs.mat"

"""dtype=[('choices', 'O'), ('free', 'O'), ('reward_left', 'O'), ('reward_right', 'O'),
          ('violations', 'O'), ('left_probability', 'O'), ('right_probability', 'O'), ('new_sess', 'O'), ('nTrials', 'O')])]],
            dtype=object)}"""

"""dict_keys(['__header__', '__version__', '__globals__', 'mousedatas'])"""

#Function to load rat data from matlab struct to python
def load_data(file):
    mat_dict = loadmat(file)
    # print(mat_dict.keys())

    # #Variables
    left_probability = [[row.flat[:] for row in line] for line in mat_dict["mousedatas"][0][0]["left_probability"]]
    # sides = [[row for row in line] for line in mat_dict["ratdata"][0][0]["sides"]]
    # rewards = [[row.flat[0] for row in line] for line in mat_dict["ratdata"][0][0]["rewards"]]
    # left_prob = [[row.flat[0] for row in line] for line in mat_dict["ratdata"][0][0]["left_prob1"]]
    # right_prob = [[row.flat[0] for row in line] for line in mat_dict["ratdata"][0][0]["right_prob1"]]
    # trial_types = [[row for row in line] for line in mat_dict["ratdata"][0][0]["trial_types"]]

    # Flatten lists
    left_probability = [item for sublist in left_probability for item in sublist]
    # sides = [item for sublist in sides for item in sublist]
    # rewards = [item for sublist in rewards for item in sublist]
    # left_prob = [item for sublist in left_prob for item in sublist]
    # right_prob = [item for sublist in right_prob for item in sublist]
    # trial_types = [item for sublist in trial_types for item in sublist]
    # print(len(left_probability[0]))

    #Convert to data frame
    df = pd.DataFrame(left_probability[0], columns=["left_probability"])
    # rat_df.columns = ["sides", "rewards", "left_prob", "right_prob", "trial_types"]
    # rat_df.apply(lambda(x))

    print(df)

    return

data = load_data(file)
