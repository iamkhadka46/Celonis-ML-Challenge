# Celonis-ML-Challenge

Gesture detection from time-series data and building production ready app
The task is to classify a given time-series into one of 8 classes, i.e., 8 different gestures. There are two aspects of this challenge: model building (from scratch and using libraries) and gathering some assumption/ideas towards making the model as a ML software to be deployed in production environment.

Dataset
I downloaded the dataset from this link. This dataset is part of the paper "uWave: Accelerometer-based personalized gesture recognition and its applications" by Jiayang Liu et al. (see more at https://www.yecl.org/publications/liu09percom.pdf). Unpacking the data, leaves several '.rar' files, having the following meanings:

Each .rar file includes gesture samples collected from one user on one day. The .rar files are named as U$userIndex ($dayIndex).rar, where $userIndex is the index of the participant from 1 to 8, and $dayIndex is the index of the day from 1 to 7.
Inside each .rar file, there are .txt files recording the time series of acceleration of each gesture. The .txt files are named as [somePrefix]$gestureIndex-$repeatIndex.txt, where $gestureIndex is the index of the gesture as in the 8-gesture vocabulary, and $repeatIndex is the index of the repetition of the same gesture pattern from 1 to 10.
In each .txt file, the first column is the x-axis acceleration, the second y-axis acceleration, and the third z-axis acceleration. The unit of the acceleration data is G (i.e., acceleration of gravity).


8 different types of gestures are shown, where the dot denotes the start and the arrow denotes the end of a gesture.

#### The two parts of the task are in files task1.py and task2.py.

#### The file model_analysis.ipynb contains scripts used to analyze the evaluate the classifier model performance.

Prerequisites:
`pip install -r requirements.txt`

#### 1. task1.py

Command to run the file:
`python task1.py`

#### 2. task2.py

Command to run the file:
`uvicorn task2:app --reload`

#### I have further deployed the app on heroku using two methods which can be accessed by the link below:

#### 1. Deployment to heroku using docker image.

- [Heroku](https://handclass-290412c88aac.herokuapp.com/)
- [SwaggerUI](https://handclass-290412c88aac.herokuapp.com/docs)

#### 2. Deployment using github actions on heroku using docker image.

- [Heroku](https://handgestureclass-31bbca3bf26e.herokuapp.com/)
- [SwaggerUI](https://handgestureclass-31bbca3bf26e.herokuapp.com/docs)
