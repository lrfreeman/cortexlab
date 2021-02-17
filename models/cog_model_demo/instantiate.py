from cog_model_classes import *

# Environment Parameters
block_flip_prob = 0.01
reward_prob_high = 0.8
reward_prob_low = 0.2

# Agent Parameters
learning_rate = 0.2
decision_noise = 10

# Instantiate environment with these parameters
environment = EnvironmentFlipsTask(block_flip_prob = block_flip_prob,
                            reward_prob_high = reward_prob_high,
                            reward_prob_low = reward_prob_low)
# Instantiate agent with these parameters
agent = AgentVanillaQ(learning_rate,
                      decision_noise)

# Experiment Parameters
n_trials = 500

# Empty variables to accumulate choices, rewards, block, qs
choices = np.zeros(n_trials)
rewards = np.zeros(n_trials)
reward_probabilities = np.zeros((n_trials,2))
qs = np.zeros((n_trials,2))

for t in range(n_trials):
  # Ask the agent for a choice
  choice = agent.get_choice()
  # Step the environment, get a reward
  reward = environment.next_trial(choice)
  # Let the agent know about the reward
  agent.update_qs(choice, reward)

  # Record the choice, reward
  choices[t] = choice
  rewards[t] = reward
  reward_probabilities[t] = environment.reward_probabilities
  qs[t] = agent.q

#---------------------------------------------------------
#Create visulisations
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)
ax1.plot(reward_probabilities)
ax1.set(title = 'Environment Reward Probabilities',
        ylabel = 'Reward Probabilities')
# plt.legend('Left', 'Right')

ax2.plot(qs)
ax2.set(title = 'Agent Q-values',
        ylabel = 'Q-values')

ax3.scatter(np.arange(n_trials), choices, 1)
ax3.set(title = 'Agent Choices',
        xlabel = 'Trial Number',
        ylabel = 'Agent Choice')

plt.show()

print('Agent Reward Rate:', np.mean(rewards))
