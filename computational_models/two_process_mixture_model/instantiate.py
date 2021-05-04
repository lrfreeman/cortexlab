from classes import *
import pandas as pd

# Environment Parameters
block_flip_prob = 0.01
reward_prob_high = 0.8
reward_prob_low = 0.2
distracter_prob = 0.2

# Agent Parameters
learning_rate_RL = 0.51
learning_rate_habit = 0.12
weight_RL = 2.76
weight_habit = 1.09
utility_reward = 1
utility_distracter = 0
utility_nothing = -1

# Instantiate environment with these parameters
environment = EnvironmentDistractersTask(block_flip_prob = block_flip_prob,
                            reward_prob_high = reward_prob_high,
                            reward_prob_low = reward_prob_low,
                            distracter_prob = distracter_prob)

# Instantiate agent with these parameters
agent = AgentKevinsModel(learning_rate_RL = learning_rate_RL,
      learning_rate_habit = learning_rate_habit,
      weight_RL = weight_RL,
      weight_habit = weight_habit,
      utility_reward = utility_reward,
      utility_distracter = utility_distracter,
      utility_nothing = utility_nothing)

# Experiment Parameters
n_trials = 500

# Empty variables to accumulate choices, rewards, block, qs
choices = np.zeros(n_trials)
rewards = np.zeros(n_trials)
distracters = np.zeros(n_trials)

reward_probabilities = np.zeros((n_trials,2))
qs = np.zeros(n_trials)
hs = np.zeros(n_trials)

for t in range(n_trials):
  # Ask the agent for a choice
  choice = agent.get_choice()
  # Step the environment, get a reward
  outcome = environment.next_trial(choice)
  # Let the agent know about the reward
  agent.learn(choice, outcome)

  # Record the choice, reward
  choices[t] = choice
  rewards[t] = outcome[0]
  distracters[t] = outcome[1]
  reward_probabilities[t] = environment.reward_probabilities
  qs[t] = agent.q
  hs[t] = agent.h

# Plot graphs
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
# ax1.plot(reward_probabilities)
# ax1.set(title="Environment Reward Probabilities", ylabel = "Reward Probabilities")
#
# ax2.plot(qs)
# ax2.plot(hs)
# ax2.set(title="Agent Q and H", ylabel = "Value")
#
# ax3.scatter(np.arange(n_trials), choices, 1)
# ax3.set(title="Agent Choices", ylabel = "Choices", xlabel = "Trial Number")

# plt.show()

print('Agent Reward Rate:', np.mean(rewards))

#Create a DF to input into GLM
def create_synthetic_data_frame():
    data_frame = pd.DataFrame([rewards, choices, distracters]).T
    #Assume choices is left choices as is behavioural data
    data_frame.columns = ["right_rewards", "left_choices", "left_rewards"]
    return(data_frame)
