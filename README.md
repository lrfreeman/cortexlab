# Cortexlab Repo

Contained within this repo is all my work created during my time at the cortexlab in UCL(https://www.ucl.ac.uk/cortexlab/). My research is supervised by the neuroscientists Matteo Carandini and Kenneth Harris. The project is partially funded by Google DeepMind (https://deepmind.com/). I'm working in close collobration with Kevin Miller (https://kevinjmiller.org/) and we are focusing on the temporal credit assignment problem. Our project explores the neural correlates of reward during a learning task in mice with the hope we can isolate computations of reward assignment. Ultimately, seeking how neurobiology can inform modern day machine learning algorithms currently used at Google.

The project is split into multiple components: __DeepLabCut__ (A opensource deep learning computer vision library) and __Ephys__ (Electrophysiology data combined with the subsequent data analysis)

## DeepLabCut (DLC)
DeepLabCut is an open source deep learning computer vision package which uses transfer learning from pre-trained neural networks to optimise the training time of your novel prediction task (https://www.nature.com/articles/s41593-018-0209-y). After manually classifying 100's of frames of videos, the DLC model can now predict animal movements automatically to the exact frame. This allows us to decide when a mouse is licking during a given trial. Important, so that we can remove licking artefacts from the neural ephys inorder to focus on the genuine neural correlates of reward. Below is a image of the software predicting when the mouse is licking.

<p align="center">
  <img width="632" alt="Screenshot 2020-09-13 at 19 06 06" src="https://user-images.githubusercontent.com/22481774/93025251-42d06380-f5f4-11ea-9c74-55b89145c3c3.png">
</p>
