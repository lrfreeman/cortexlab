# Cortexlab Repo

This github repo is a collection of projects undertaken whilst studying neuroscience at the CortexLab at UCL with Matteo Carandini and Kenneth Harris (https://www.ucl.ac.uk/cortexlab/). My work is a collaboration with my supervisor Kevin Miller, a post-doc at UCL and a research scientist at Google DeepMind (https://deepmind.com/).

My research explores the temporal credit assignment problem in the field of reinforcement learning. As rewards can occur in a temporally delayed fashion, this causes the problem of determining the actions that lead to a certain outcome (Minsky, 1961; https://courses.csail.mit.edu/6.803/pdf/steps.pdf). By exploring the underlying neural activity of a biological agent during a two-armed bandit task, we hope to use signal processing techniques to isolate the computations of reward assignment. Ultimately, my goal is to explore how neurobiology can inform modern day machine learning algorithms.

The project is split into multiple components: __deepLabCut__ (An open source deep learning computer vision library I've added to in order to classify tongue kinematics for the purpose of removing signal artefacts during electrophysiological analysis), __electrophysiology__ (a set of scripts converting neural data into a python arrays) and finally, __computational_models__ a mix of Q-Learning algorithms and regression models (for the purposes of signal processing and cognitive behavioural modelling).

All of the data below is based on electrophysiological recordings from Neuropixels (https://www.ucl.ac.uk/neuropixels/), 1000 count probes. Combined with behavioural data from rodents conducting a two-armed bandit task.

<p align="center">
  <img width="1398" alt="Screenshot 2020-12-03 at 11 02 18" src="https://user-images.githubusercontent.com/22481774/117122762-20ab7180-ad8e-11eb-8037-703933ab3394.png">
</p>

## Task explanation
During the task a mouse can receive four reward permutations (A cherry reward, a grape reward, both rewards at the same time and no rewards). The mouse is head fixed, and can make two choices on a steering wheel. Given the correct choice, a reward is delivered via a spout to the mouse. One of the reward's is random (Grape - a "distractor" reward) whereas another reward is released on a successful trial (Cherry). Thus neurally, we hypothesis that the mouse should care more about cherry than grape, that this preference will be represented neurally, and that these signals can be analysed to understand the neurobiological underpinnings of credit assignment.

<p align="center">
  <img width="346" alt="Screenshot 2020-12-03 at 11 02 18" src="https://user-images.githubusercontent.com/22481774/117124811-ae885c00-ad90-11eb-8493-5de16a0fa3a2.png">
</p>

## Signal Processing - Basic
Given that our biological agent licks to receive a reward on completion of a successful trial, a signal processing problem is presented. Specifically, an inability of decoding whether a neuron is responding to a given reward or whether a neuron is firing as a result of tongue movements. By looking at the below peri-stimulus time histogram and a spike-raster that I've generated, the visualisations show that specific cells may respond differently to different reward permutations.

<p align="center">
  <img width="1398" alt="Screenshot 2020-12-03 at 11 02 18" src="https://user-images.githubusercontent.com/22481774/117123421-e55d7280-ad8e-11eb-823e-eac3463194f0.png">
</p>

However, how do we know that these signals are not artefacts of licking? Perhaps the mouse really likes licking. Or licks more during a cherry reward than a grape reward. Which begs the question, how can we decode whether a neuron is firing because a reward touched the tongue, or because the tongue has moved out of the mouth?

## DeepLabCut (DLC)
In order to prevent introducing further electrical artefacts via an electrical tongue measuring device, a computer vision approach was selected to measure tongue kinematics. For this I used DeepLabCut, an open source deep learning computer vision package which uses transfer learning from pre-trained neural networks to optimise the training time of your novel prediction task (https://www.nature.com/articles/s41593-018-0209-y). I placed a server on AWS running DLC on a GPU instance for speed and wrote a python application to predict when and where the mouse is licking. The ultimate output of this project was a list of licking times for each behavioural session.

## Signal Processing - More advanced
A kernel regression model was used to analyse different events (Licking vs reward times) to understand which event had the greatest affect on spike count. Specifically, I implemented the below model from Parker et al,. (2016):

<p align="center">
  <img width="1398" alt="Screenshot 2020-12-03 at 11 02 18" src="https://user-images.githubusercontent.com/22481774/117127251-c2818d00-ad93-11eb-8102-db09b03573ea.png">
</p>

Where S(t) is spikes at time t. Before creating this model I checked that the covariance between each event was weak to ensure the model will work.

<p align="center">
  <img width="1398" alt="Screenshot 2020-12-03 at 11 02 18" src="https://user-images.githubusercontent.com/22481774/117127602-31f77c80-ad94-11eb-9901-83dd6ba75ead.png">
</p>
