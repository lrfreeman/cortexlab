# Cortexlab Repo

This github repo is a collection of projects undertaken whilst studying neuroscience at the CortexLab at UCL (https://www.ucl.ac.uk/cortexlab/) with Matteo Carandini and Kenneth Harris. My work is a collaboration with my supervisor Kevin Miller, a post-doc at UCL and a research scientist at Google DeepMind (https://deepmind.com/).

The focus of my research is on solving the temporal credit assignment problem in the field of reinforcement learning. Rewards can occur in a temporally delayed fashion. This causes the problem of determining the actions that lead to a certain outcome (Minsky, 1961; https://courses.csail.mit.edu/6.803/pdf/steps.pdf)


Our project explores the neural correlates of reward during a learning task in mice with the hope we can isolate computations of reward assignment. Ultimately, seeking how neurobiology can inform modern day machine learning algorithms currently used at Google.

The project is split into multiple components: __DeepLabCut__ (An open source deep learning computer vision library) and __Ephys__ (Electrophysiology data combined with the subsequent data analysis)

## DeepLabCut (DLC)
DeepLabCut is an open source deep learning computer vision package which uses transfer learning from pre-trained neural networks to optimise the training time of your novel prediction task (https://www.nature.com/articles/s41593-018-0209-y). After I manually classified 100's of frames of videos, the resulting DLC model can now predict licking kinematics. This allows us to decide when a mouse is licking during a given trial. Important, so that we can remove licking artefacts from the neural ephys in order to focus on the genuine neural correlates of reward.

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
