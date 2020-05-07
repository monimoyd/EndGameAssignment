# EndGameProject
EngGameAssignment

# I. Project Overview
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


# II. Environment
The agent i.e. car takes from the current state an action based on the policy and environment tells the next state and gives a reward.
a.	Observation State Space:
The following a few picture samples show the examples of one part of observation state space

![sand1](/doc_images/sand_1.png)   ![sand2](/doc_images/sand_2.png)     ![sand3](/doc_images/sand_3.png) 
![sand4](/doc_images/sand_4.png)   ![sand5](/doc_images/sand_5.png)     ![sand6](/doc_images/sand_6.png) 

Observation state space consists of a image 
-	Taking mask value of sand by taking 80x80 numpy array by taking 40 pixel around the car.
-	 A isosceles triangular shaped car (i.e. only two sides are equal ) rotated in the direction the car is moving is superimposed on the image. Isosceles triangle help in asymmetry
-	 A numbered score representing what the car did in this step is also superimposed. Numbered scores are represented as per below table:

| Number  | What the car did                                                                     |
| ------- | ------------------------------------------------------------------------------------ |
|   0     |  Car hit the boundary                                                                |
|   1     |  Car has done 360 degree rotation clockwise/counter clockwise                        |
|   2     |  Car is off the Road                                                                 |
|   3     |  Car is on the Road but distance to destination is increased from the last position  |
|   4     |  Car is on the Road but distance to destination is reduce from the last position     |
|   5     | Car has reached the Destination goal                                                 |


This image is processed by CNN and LSTM
In addition the following additional attributes are used in state space



| Number          | What the car did                                                                                                   |
| ----------------| ------------------------------------------------------------------------------------------------------------------ |
| Car Angle       | The angle the car is rotated divided by 360. If angle is more than 360 or less than -360, modulus operation is done|  
| Orientation     |  Angle of orientation of current position of car to the destination goal divided by 360                            |
| On Road         | Whether car is on road, It has value 1 if car is on road, 0 if car is off road                                     |     |Diff of Distance | Difference of distance of the car to the Destination goal from the current position and the last position          |
|                 | normalized by dividing by 4








