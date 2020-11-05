#Steps
ssh -i ~/.ssh/cortexlab.pem ubuntu@3.14.126.254
source activate DLC-GPU
cd deeplabcut
ipython
import DeepLabCut

#Set config to variable for reuse
config_path = '/home/ubuntu/DeepLabCut/Master_Project-L-2020-08-13/config.yaml'

# Use frames from differnet behavioural sessions, different lighting, different animals
# For simple behaviours 100-200frames gives good results
# Keep frame size small to reduce training times
# Set crop size to true then you'll be asked to draw a white box around the GUI which will written to config file
# Choose frames from when the behaviour happens - do so by kmeans
deeplabcut.extract_frames(config_path, mode='automatic/manual', algo='uniform/kmeans', userfeedback=False, crop=True/False)
deeplabcut.check_labels(config_path, visualizeindividuals=True/False)

#"""List of functions to call in ipthon on AWS"""
config_path = '/home/ubuntu/DeepLabCut/TAWS-Laurence-2020-07-30/config.yaml'
deeplabcut.create_training_dataset(config_path)
deeplabcut.cropimagesandlabels(config_path)

#It is recommended to train the ResNets for 200k of iterations until the loss plateaus (typically around 200,000) if you use batch size 1
deeplabcut.train_network(config_path, gputouse=0, displayiters=100, saveiters=1500, maxiters=50000)
deeplabcut.evaluate_network(config_path,Shuffles=[1], plotting=True, gputouse=0)
deeplabcut.analyze_videos(config_path, ['/home/ubuntu/DeepLabCut/TAWS-Laurence-2020-07-30/videos/C_sample_outcome_movie_both.avi'],save_as_csv=True)
deeplabcut.analyze_videos(config_path, ['/home/ubuntu/DeepLabCut/Master_Project-L-2020-08-13/videos/video_snippet_KM011_2020-03-19_trial99.avi'],save_as_csv=True)

#You can pass trailpoints to see history of points
deeplabcut.create_labeled_video(config_path,['/home/ubuntu/DeepLabCut/TAWS-Laurence-2020-07-30/videos/C_sample_outcome_movie_both.avi'], trailpoints=10)
deeplabcut.create_labeled_video(config_path,['/home/ubuntu/DeepLabCut/Master_Project-L-2020-08-13/videos/video_snippet_KM011_2020-03-19_trial99.avi'])

#Plot-poses is parts vs time
deeplabcut.plot_trajectories(config_path, [‘fullpath/analysis/project/videos/reachingvideo1.avi’])
deeplabcut.plot_trajectories(config_path,['/home/ubuntu/DeepLabCut/Master_Project-L-2020-08-13/videos/video_snippet_KM011_2020-03-19_trial98.avi'])

#"""Config file changes"""
alpha value = #Transparency of plotted labels
numframes2pick = #May need to increase to reach recommended 100-200

#Label FPS
ffmpeg -i /Users/laurence/Desktop/Trial_400_200Kiter.mp4 -vf "drawtext=fontfile=Arial.ttf: text=%{n}: x=(w-tw)/2: y=h-(2*lh): fontcolor=white: box=1: boxcolor=0x00000099" -y output.mp4
pwd

DLC

git clone https://github.com/AlexEMG/DeepLabCut.git
conda env create -f DLC-GPU.yaml (CD into conda-enviorments)
export DLClight=True
ipython
import deeplabcut
python test.script

Misc

#ssh -i ~/.ssh/cortexlab.pem ubuntu@18.216.113.119
Nvcc --version
nvidia-smi
Jupyter nbconvert --execute flie.ipynb
/usr/local/cuda -> /usr/local/cuda-10.0
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.0 /usr/local/cuda

Create partition

sudo dmesg | grep xvdb
df -h
sudo fdisk /dev/xvdb
sudo mkfs.ext4 /dev/
cd
ls
mkdir payload
ls
sudo mount /dev/xvdb1 payload
df -h
sudo chmod a+rwx payload/

Links

+ https://stackoverflow.com/questions/30820513/what-is-the-correct-version-of-cuda-for-my-nvidia-driver/30820690#30820690
+ https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-base.html
