# EndGameProject
EngGameAssignment

# Project Overview
Problem Statement

![Kivy Car Environment](/doc_images/kivy_car_environment.png)

For this project a map of a city with the roads is provided as an image. Also a car is provided. My task is:

1.	To keep the car on the road maximum number of times
2.	To reach goal in minimum number of time steps. 
I have used Kivy Environment for this project
https://kivy.org/doc/stable/installation/installation-windows.html with maps provided by TSAI.
In this project I have created a framework for Deep Reinforcement Learning.

Highlights of various activities I have done in the project
-	Created a simulated gym environment based on Kivy environment 
-	Used Twin Delayed Deep Deterministic (TD3) algorithm  for training
-	Used LSTM and Convolutional Neural Network(CNN) based DNN model
-	For state space, mask is superimposed with a triangular rotated car  along with a numbered score based on what the car did in the step
-	Random sampling of actions initially to fill up the replay buffer initially
-	Mixture of random actions and policy actions are done based on exploration factor epsilon value which is reduced every episode
-	After every two training, evaluation is done 
-	Metrics (Total Rewards, On Road/Off Road count, Goal Hit Count, Boundary Hit Count) are calculated in each episode of training as well as evaluation
-	Both Actor and Critic Models after each episode is stored
-	Based on Analytics  on collected Metrics for evaluation and training episodes I have chosen appropriate model for testing
-	Tuned parameters like learning rate, weight decay to overcome agent circling  (i.e. Ghoomar) effect

How to install and Run on Windows:
i.	First create a new conda environment
conda create -n rl python=3.7 anaconda
conda activate rl

ii.	If you have only CPU: 

conda install -c pytorch pytorch-cpu

If you have GPU make sure that CUDA, CUDNN and drivers are already installed

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch



iii. Install kivy using command below:

conda install -c conda-forge kivy

iv.	Clone the repository

git clone https://github.com/monimoyd/EndGameAssignment.git

v.	Training

For training use the command below

python TD3_train.py

vi.	Testing 

For Testing use the command below:

python TD3_test.py




