# Cortexlab Repo

This github repo is a collection of projects undertaken whilst studying neuroscience at the CortexLab at UCL with Matteo Carandini and Kenneth Harris (https://www.ucl.ac.uk/cortexlab/). My work is a collaboration with my supervisor Kevin Miller, a post-doc at UCL and a research scientist at Google DeepMind (https://deepmind.com/).

My research explores the temporal credit assignment problem in the field of reinforcement learning. As rewards can occur in a temporally delayed fashion, this causes the problem of determining the actions that lead to a certain outcome (Minsky, 1961; https://courses.csail.mit.edu/6.803/pdf/steps.pdf). By exploring the underlying neural activity of a biological agent during a two-armed bandit task, we hope to use signal processing techniques to isolate the computations of reward assignment. Ultimately, my goal is to explore how neurobiology can inform modern day machine learning algorithms.

The project is split into multiple components: __deepLabCut__ (An open source deep learning computer vision library I've added to in order to classify tongue kinematics for the purpose of removing signal artefacts during electrophysiological analysis), __electrophysiology__ (a set of scripts converting neural data into a python arrays) and finally, __computational_models__ a mix of Q-Learning algorithms and regression models (for the purposes of signal processing and cognitive behavioural modelling).

All of the data below is based on electrophysiological recordings from Neuropixels (https://www.ucl.ac.uk/neuropixels/), 1000 count probes. Combined with behavioural data from rodents conducting a two-armed bandit task.

<p align="center">
  <img width="1398" alt="Screenshot 2020-12-03 at 11 02 18" src="https://user-images.githubusercontent.com/22481774/117122762-20ab7180-ad8e-11eb-8037-703933ab3394.png">
</p>

## Task explanation
During the task a mouse can receive four reward permutations (A cherry reward, a grape reward, both rewards at the same time and no rewards). The mouse is head fixed, and can make two choices on a steering wheel. Given the correct choice, a reward is delivered via a spout to the mouse.

<p align="center">
  <img width="346" alt="Screenshot 2020-12-03 at 11 02 18" src="https://user-images.githubusercontent.com/22481774/117124811-ae885c00-ad90-11eb-8493-5de16a0fa3a2.png">
</p>

## Signal Processing
Given that our biological agent licks to receive a reward on completion of a successful trial during a two-armed bandit task a signal processing problem is presented. Specifically, an inability of decoding whether a neuron is responding to a given reward or whether a neuron is firing as a result of tongue movements.

Reward permutation explanation: During the task a mouse can receive four reward permutations (A cherry reward, a grape reward, both rewards at the same time and no rewards). Given the reward permutations that occur, can we find neural correlates of the reward within the Neuropixel data. The first thing I did was generated a peri-stimulus time histogram and a spike-raster to explore whether the different reward types are represented neurally.

<p align="center">
  <img width="1398" alt="Screenshot 2020-12-03 at 11 02 18" src="https://user-images.githubusercontent.com/22481774/117123421-e55d7280-ad8e-11eb-823e-eac3463194f0.png">
</p>

## DeepLabCut (DLC)
DeepLabCut is an open source deep learning computer vision package which uses transfer learning from pre-trained neural networks to optimise the training time of your novel prediction task (https://www.nature.com/articles/s41593-018-0209-y). After I manually classified 100's of frames of videos, the resulting DLC model can now predict licking kinematics. This allows us to decide when a mouse is licking during a given trial. Important, so that we can remove licking artefacts from the neural data in order to focus on the genuine neural correlates of reward.

The ultimate outcome of creating this library was to create a signal processing technique to

## Ephys
Python software that inputs kilosorted electrophysiology data, manipulates the data and generates data visulisations such as peristimulus time histograms across 100's of trials of neuronal spikes. The data is collected through Neuropixel probes.

The below chart shows 5 subplots. The first is the PSTH associated with a single cluster of cells. The other four, are the predicted licking kinematics of the mouse, answering the question what reward spout is the mouse licking and when.

<p align="center">
  <img width="1398" alt="Screenshot 2020-12-03 at 11 02 18" src="https://user-images.githubusercontent.com/22481774/101001445-0d824480-3557-11eb-8337-b8f30988bb99.png">
</p>

The below chart is a spike raster visulisation produced in python with each trial type coloured differently.

<p align="center">
  <img width="1398" alt="Screenshot 2020-12-03 at 11 02 18" src="https://user-images.githubusercontent.com/22481774/101002818-0445a780-3558-11eb-9fa7-6eea890db888.png">
</p>
