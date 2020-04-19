# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./class_distribution.png "Visualization"

[image4]: ./test_images/test1_17.jpg "Traffic Sign 1"
[image5]: ./test_images/test2_38.jpg "Traffic Sign 2"
[image6]: ./test_images/test3_12.jpg "Traffic Sign 3"
[image7]: ./test_images/test5_13.jpg "Traffic Sign 4"
[image8]: ./test_images/test6_33.jpg "Traffic Sign 5"
[image9]: ./test_images/test7_1.jpg "Traffic Sign 6"
[image10]: ./test_images/test8_18.jpg "Traffic Sign 7"
[image11]: ./test_images/multi1_33.jpg "Traffic Sign 8"
[image12]: ./test_images/multi1_13.jpg "Traffic Sign 9"
[image13]: ./test_images/challenge3_28.jpg "Traffic Sign 10"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410 images
* The size of test set is 12630 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here I plotted the distribution of classes in a histogram for each of the train, validate, and test datasets. Here I calculated the class with the most images. In this case, class 2 had the most number of occurrences in the training set. It is likely that the model will be better at predicting this class over others with fewer occurrences in the training data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried two different preprocessing techniques. The first was to simply normalize the pixel data between 0-1. This is for better model handling. I also attempted to grayscale the image. As discussed later in the writeup, I did not observe significant improvements over full color images, so I reverted this change so my preprocessing only normalizes the image.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model closely follows the LeNet architecture used in the lab - a deep neural net with two convolution layers and three fully connected layers. The differences are described below where I describe my approach to improve the model for this task.

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Dropout   |   |
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32  |
| RELU					|												|
| Dropout   | |
| Max pooling   | 2x2 stride, outputs 5x5x32  |
| Flatten   |   |
| Fully connected		| 800 to 400	|
| RELU					|												|
| Dropout   | |   
| Fully connected   | 400 to 120   |
| RELU					|												|
| Dropout   | |   
| Fully connected | 120 to 43 |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the `AdamOptimizer`. This seems to be an effective method for training in the lab, so I focused on preprocessing and model architecture changes to optimize this model for this task.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 96.3%
* test set accuracy of 95.8%

If an iterative approach was chosen:
* First, I started with the LeNet architecture exactly as it was in the lab, with a few adjustments to the dimensions to accept the 3 channel image and the 43 class output. My goal was to make sure my image preprocessing was working and that the pipeline could achieve the approximate 89% accuracy. Once I reached this, accuracy, I moved on to start improving the model.
* Next, I added dropout after each layer (see the architecture layout table). This is a known technique that can be used to reduce overfitting. The "Test Parameter Tracker" table in the notebook shows the parameters used and the test accuracy. I was able to achieve 93.4% accuracy with 50% dropout. This gave some improvement, but it performed worse on the additional images for testing, so I believe it was slightly overfit. Exploring other architectures and techniques is necessary to obtain a more robust model. Regardless, dropout is important for this task because, when dealing with images, it helps distribute the weights between all parameters. Practically, this means that not one characteristic is relied upon to identify a class. For example, if part of the sign is blocked by the tree, it's shape isn't the only characteristic it is using to identify it.
* I then tried a model with slightly more features, increasing the second convolution layer to 24 kernels and resizing the fully connected layers after it. This resulted in marginally better performance.
* According to [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) linked in the notebook, grayscale images performed slightly better than full rgb images. I next converted the data set to grayscale and observed similar performance.
* Since grayscale had minimal improvement, I wanted to first optimize the model with a full color image. Reverting back to a 3 channel image changes, I incrementally increased the size of the convolution kernels and fully connected layers. This increased the test performance by ~ 3%.

If a well known architecture was chosen:
* What architecture was chosen and why is it relevant to this application? I chose this architecture because to serve as a starting point. From the lab, we have seen it is capable of classifying 10 different objects in images. It would follow that it could classify more so long as it was scaled properly. Scaling this model is a good way to explore the importance of model sizing on over/under fitting while still learning about the value of preprocessing and hyperparameters for CNN model design.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? Validation accuracy is a helpful measure of the model over or under fitting. As I tuned the parameters, I looked for the validation accuracy to start to diminish, indicating it had maximized its fit and was beginning to overfit. The training accuracy was not used to measure the performance, but that should likely be the maximum accuracy during training. The test accuracy is helpful to determine its performance. The addition of new images is another important test of the model. Since the training data likely has some uniformities (sign scale, shape, coloring, position/angle in the image), it is important to test a completely separate dataset to observe the performance in novel conditions. Of course, preprocessing can help with this by generating random scaling, angles, translations, random color/brightness adjustments.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I found 6 unique traffic sign images on the web. One of the images has two signs, so I split it up into four possible images. The first two as each sign cropped so that there is only one in the frame. The second two are the image with both signs classified as one or the other correct classifications. My intent was to observe how the system performs when two learned signs are present in the frame. Hopefully, the model will pick one of them correctly. Upon further inspection of the softmax probabilities, I expect to see some distribution of the probability between both of the classes.  
<img src="./test_images/test2_38.jpg" width="150"/>
<img src="./test_images/test3_12.jpg" width="150"/>
<img src="./test_images/test5_13.jpg" width="150"/>
<img src="./test_images/test6_33.jpg" width="150"/>
<img src="./test_images/test7_1.jpg" width="150"/>
<img src="./test_images/test8_18.jpg" width="150"/>
<img src="./test_images/multi1_33.jpg" width="150"/>
<img src="./test_images/multi1_13.jpg" width="150"/>
<img src="./test_images/challenge3_28.jpg" width="150"/>
<img src="./test_images/test1_17.jpg" width="150"/>

The last image in the array was considered a "challenge" image, since it is a common shape with blurry features. There are multiple classes with the red triangle as the frame for different signs (see the "!" image for example), so it will be interesting to see how the model performs on this image - especially since the image is blurry and not easy for even humans to distinguish.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Keep Right      		|  Keep Right									|
| Priority Road    | Priority Road |
| Yield					| Speed Limit 70 |
| Yield/Turn Right Ahead   		| 30 KPH |
| Turn Right Ahead  | Turn Right Ahead  |
| 30 KPH		|	30 KPH				|
| Children Crossing   | Priority Road |
| No Entry   | No Entry  |
| General Caution   |  General Caution |


The model was able to correctly guess 6 of the 9 traffic signs, which gives an accuracy of 66.7%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The below image was generated from the notebook, where I plot the top 5 softmax probabilities along with the image it was attempted to classify on.  
<img src="./softmax_predictions.png"/>

For the image with two signs, I was not able to identify either of the signs in the image. There may be a few reasons for this. First, the signs may be too small. Perhaps if I used a scaling distortion on the training data, we could have better performance across different size signs. I do notice that the yield class is the second most likely, which indicates that the model is behaving somewhat correctly. I think this case brings up an interesting real world use case, where multiple signs might be in the view of a vehicles camera system. Perhaps some preprocessing would identify signs and crop the image first around the signs before passing them to the model to prevent multiple signs in one input.

Regarding the yeild sign that was incorrectly identified. It could be possible that that the color of the sign (red border with white interior) is causing the sign to be identified as a speed limit sign. This, along with the fact that some of these predictions are "100%" indicates that the model is overfit for some of these signs. This exploration of softmax probabilities has been helpful to qualify the performance of the model.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
