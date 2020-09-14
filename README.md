# Cortexlab
A repo for all my work created during my time at the cortex lab in UCL. 

My research is supervised by Matteo Carandini and Kenneth Harris of the Cortexlab at UCL. The project is partially funded by DeepMind and focuses on the temporal credit assignment problem. We are exploring neural correlates of reward during a learning task in mice with the hope we can isolate computations of reward assignment. Ultimately, seeking how neurobiology can inform modern day machine learning algorithms currently used at Google.

Attached are a few scripts.

Is_Licking - Takes in the output from Deep Lab Cut and applies additional logic to ensure the model can predict when the mouse is licking to a 99% accuracy.

Process_data - Takes in the raw CSV output from Deep Lab Cut and Processes it for the Is_Licking script

Ingest_Time_Sync - Takes in a combination of kilosorted eletrophysiology data and the output from Is_Licking in order to remove licking artefacts from the physiology data inorder to focus on the neural correlates of reward. 

<img width="632" alt="Screenshot 2020-09-13 at 19 06 06" src="https://user-images.githubusercontent.com/22481774/93025251-42d06380-f5f4-11ea-9c74-55b89145c3c3.png">
