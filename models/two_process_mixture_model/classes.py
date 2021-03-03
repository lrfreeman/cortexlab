import numpy as np
import matplotlib.pyplot as plt

class EnvironmentDistractersTask:
    # An environment for a the distracter rewards task

    def __init__(self, block_flip_prob, reward_prob_high, reward_prob_low, distracter_prob):
        # Assign the input parameters as properties
        self._block_flip_prob = block_flip_prob
        self._reward_prob_high = reward_prob_high
        self._reward_prob_low = reward_prob_low
        self._distracter_prob = distracter_prob

        # Choose a random block to start in
        self._block = np.random.binomial(1, 0.5)

        # Set up the new block
        self.new_block()

    def new_block(self):
        # Flip the block
        self._block = 1 - self._block
        # Set the reward probabilites
        if self._block == 1:
          self.reward_probabilities = [self._reward_prob_high,
                                       self._reward_prob_low]
        else:
          self.reward_probabilities = [self._reward_prob_low,
                                       self._reward_prob_high]


    def next_trial(self, choice):
        # Choose the reward probability associated with the choice that the agent made
        reward_prob_trial = self.reward_probabilities[choice]

        # Sample a reward with this probability
        reward = np.random.binomial(1, reward_prob_trial)
        # Sample a distracter
        distracter = np.random.binomial(1, self._distracter_prob)

        # Check whether to flip the block
        if np.random.binomial(1, self._block_flip_prob):
          self.new_block()

        # Return the reward
        return reward, distracter

class AgentKevinsModel:
# The agent from Kevin's Data Club talk
    def __init__(
          self,
          learning_rate_RL,
          learning_rate_habit,
          weight_RL,
          weight_habit,
          utility_reward,
          utility_distracter,
          utility_nothing):

        self._learning_rate_RL = learning_rate_RL
        self._learning_rate_habit = learning_rate_habit
        self._weight_RL = weight_RL
        self._weight_habit = weight_habit
        self._utility_reward = utility_reward
        self._utility_distracter = utility_distracter
        self._utility_nothing = utility_nothing

        # Initialize Q and H to 0
        self.q = 0
        self.h = 0

    def softmax(self, x):
        softmax_x = np.exp(x) / np.sum(np.exp(x))
        return softmax_x

    def get_choice(self):
        # Draw choice probabilities according to the softmax
        logit = self._weight_RL * self.q + self._weight_habit * self.h
        choice_probs = self.softmax([logit, -1*logit])
        # Select a choice according to the probabilities
        choice = np.random.binomial(1, choice_probs[0])

        return choice

    def learn(self, choice, outcome):
        reward = outcome[0]
        distracter = outcome[1]

        if reward & ~distracter:
          trial_utility = self._utility_reward
        elif ~reward & distracter:
          trial_utility = self._utility_distracter
        elif reward & distracter:
          trial_utility = 1
        elif ~reward & ~distracter:
          trial_utility = self._utility_nothing
        else:
          raise ValueError()

        # Convert choice from 0 or 1 to -1 or +1
        if choice == 1:
          choice_for_learn = 1
        elif choice == 0:
          choice_for_learn = -1
        else:
          raise ValueError()

        # Update the q-value of the chosen side towards the reward
        self.q = self.q * (1 - self._learning_rate_RL) + trial_utility * choice_for_learn * self._learning_rate_RL
        self.h = self.h * (1 - self._learning_rate_habit) + choice_for_learn * self._learning_rate_habit
