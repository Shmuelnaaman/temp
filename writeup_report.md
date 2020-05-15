# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image7]: ./samples.png "Sample images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

I have modified my drive.py, so that the car goes faster (30).

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a modified version of the Nvidia network from the video in section 15. of the course. See more details, below.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py line 94, 97). 

The model was trained and validated on generated data sets to ensure that the model was not overfitting (code from line 18). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 94, 97).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road via generating data from the left right camera images.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one presented in the course I thought this model might be appropriate because it was created by NVIDIA for similar purposes.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it contains dropouts. 

Then I cropped the images to only show the relevant part, where the road is. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I have recorded additional images with the simulator around those areas.

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 80-100) consisted of a convolution neural network with the following layers and layer sizes.

Conv2D(24, (5 5), strides=(2, 2), activation="elu"))
Conv2D(36,(5, 5), strides=(2, 2), activation="elu"))
Conv2D(48 (5, 5), strides=(2, 2), activation="elu"))
Conv2D(64, (3, 3), activation="elu"))
Conv2D(64, (3, 3), activation="elu"))
Flatten())
Dense(100))
Dropout(0.5))
Dense(50))
Dropout(0.5))
Dense(10))
Dense(1))

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a lap on the track one using center lane driving, as far as I could. 
To augment the data set, I also flipped images and angles thinking that this would simualte recovery from the sides:

![alt text][image7]

I finally randomly shuffled the data set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by. I used an adam optimizer so that manually training the learning rate wasn't necessary:

Train on 10930 samples, validate on 1215 samples
Epoch 1/6
10930/10930 - 33s 3ms/step - loss: 0.0355 - val_loss: 0.0166
Epoch 2/6
10930/10930 - 24s 2ms/step - loss: 0.0169 - val_loss: 0.0139
Epoch 3/6
10930/10930 - 25s 2ms/step - loss: 0.0134 - val_loss: 0.0130
Epoch 4/6
10930/10930 - 26s 2ms/step - loss: 0.0122 - val_loss: 0.0123
Epoch 5/6
10930/10930 - 25s 2ms/step - loss: 0.0114 - val_loss: 0.0117
Epoch 6/6
10930/10930 - 25s 2ms/step - loss: 0.0110 - val_loss: 0.0127
