import numpy as np
import matplotlib.pyplot as plt

class EnvironmentFlipsTask:
  # An environment for a very simple two-armed bandit RL task

  def __init__(self, block_flip_prob, reward_prob_high, reward_prob_low):
    # Assign the input parameters as properties
    self._block_flip_prob = block_flip_prob
    self._reward_prob_high = reward_prob_high
    self._reward_prob_low = reward_prob_low

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

    # Check whether to flip the block
    if np.random.binomial(1, self._block_flip_prob):
      self.new_block()

    # Return the reward
    return reward

class AgentVanillaQ:
  # An agent implementing textbook Q-learning, with a softmax policy

  def __init__(self, positive_learning_rate, negative_learning_rate, decision_noise):
    self._positive_learning_rate = positive_learning_rate
    self._negative_learning_rate = negative_learning_rate
    self._decision_noise = decision_noise
    # Initialize q-values to 50%
    self.q = 0.5 * np.ones(2)

  def softmax(self, x):
    softmax_x = np.exp(x) / np.sum(np.exp(x))
    return softmax_x

  def get_choice(self):
    # Draw choice probabilities according to the softmax
    choice_probs = self.softmax(self._decision_noise * self.q)
    # Select a choice according to the probabilities
    choice = np.random.binomial(1, choice_probs[1])
    return choice

  def update_qs(self, choice, reward):
      #Positive Prediction Error
      if reward - self.q[choice] > 0:
          # Update the q-value of the chosen side towards the reward using a positive learning rate
          self.q[choice] = self.q[choice] * (1 - self._positive_learning_rate) + self._positive_learning_rate * reward
      #Negative Prediction Error
      elif reward - self.q[choice] < 0:
          # Update the q-value of the chosen side towards the reward using a negative learning rate
          self.q[choice] = self.q[choice] * (1 - self._negative_learning_rate) + self._negative_learning_rate * reward
      else: print("No prediction Error")
