# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test1.jpg"
[image5]: ./test2.jpg"
[image6]: ./test3.jpg"
[image7]: ./test4.jpg"
[image8]: ./test5.jpg"
[image9]: ./Train_Dist.jpg"
[image10]: ./Test_Dist.jpg"
[image11]: ./Validation_Dist.jpg"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how training, testing and validation data is distributed regarding number of images belonging to each class

![alt text][image9]
![alt text][image10]
![alt text][image11]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to leave the images to be colored and not grayscale because I wanted to keep the information encapsulated in the 3 RGB channels ...

I normalized the image data because Deep learning models perform better on normalized data so I devided all the images by 255 to normalize the input to be between 0 and 1

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x289x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten    	      	| input 5x5x16,  outputs 400x1    				|
| Fully connected   	| input 400, output 120      					|
| RELU					|												|
| Fully connected   	| input 120, output 84      					|
| RELU					|												|
| Output Fully connected| input 84, output 43       					|
| Softmax				| convert to probs        						|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer as it is known to be better than normal SGD as it updates the learning rate accordingly to ensure descending to the global minima. for the batch size I used 128 batch size, I din't want to increase it to prevent memory issues and it gave pretty good results. for the number of epochs I trained the model for 30 epochs but I was not satisfied by the output so I trained it again for 60 epochs which gave me good results. For the learning rate I chose the default 0.001 value.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 94.7%
* test set accuracy of 93.365%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
- I started with a simple architecture that consists of one 5x5 convulotional layer(3 filters) followed by 2 fully connected layers (first one outputs 84 and second one outputs 43 neurons) . I wanted to experiment to see its performance so I chose this model.

* What were some problems with the initial architecture?

- This model gave a validation accuracy of around 70% and testing accuracy of 83% even after increasing the number of epochs the accuracy didn't increase much so I decided to increase the number of layers to make the model be able to detect more features.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

- It seemed like the obvious choice to increase the layers in the model to make it able to detect more features and avoid overfitting as the training and validation accuracies were not good enough. so I added an additional 5x5 convolutional layers so that the first one with 6 filters and the second one with 16 filters and in each one a relu activation function was used. Each of these 2 convolutional layers were followed by a 2x2 max pooling layer. I added an additional fully connected layer after flattening the convolutional layers output which takes a flattened vector of 400 values and outputs 120 values (neurons) 

* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

- I tried tuning the number of filters of each convolutional layers so for the first layer I started with 3 filters and for the second layer I started with 6 filters. After training the model its validation accuracy increased but still was not good enough so I increased the number of filters such that the first layer is of 6 filters and the second one is of 16 filters. I also experimented with changing the number of output neurons from the added fully connected layer unitl I was satisfied with the selected value (120). I also tried tuning the learning rate so I changed it to 0.1 but found the validation accuracy to be oscillating as it caused it to increase/decrase quite randomly so after changing it to the default value 0.001 this problem was fixed.

I think adding dropout layers would have helped the model to regularize more and avoid overfitting.

If a well known architecture was chosen:
* What architecture was chosen?
I used the LeNet architecture.

* Why did you believe it would be relevant to the traffic sign application?
This model is proven to work well on traffic signs as after doing some research I found many papers using this architecture for the task at hand. 
https://ieeexplore.ieee.org/document/8652024

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The model's accuracy on the training set is 100% which made me worry that it might be overfitting but I found that the validation accuracy is 94.7% and testing accuracy is 93.365% which made me satisfied with the model's performance.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second and fourth images might be difficult to classify because it has many trees on the background which might cause the model to get confused.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| Yield     			| Yield 										|
| Bumpy road			| Beware of ice/snow							|
| Speed limit (30km/h)	| Speed limit (30km/h)      	 				|
| Keep left 			| Keep left         							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is sure that this is a No entry sign (probability of 1). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| No Entry   									| 
| 0     				| Speed limit (30km/h) 							|
| 0  					| Stop               							|
| 0  	      			| Priority Road  		    			 		|
| 0 				    | Yield          						     	|


For the second image ... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Yield   								    	| 
| 0     				| Speed limit (50km/h) 							|
| 0  					| Bicycles crossing               				|
| 0  	      			| Road work  		         			 		|
| 0 				    | Children crossing						     	|

For the Third image ... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Beware of ice/snow   							| 
| 0     				| Bicycles crossing 							|
| 0  					| Children crossing             				|
| 0  	      			| Slippery road  		      			 		|
| 0 				    | Wild animals crossing						    |

For the Fourth image ... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Speed limit (30km/h)  						| 
| 0     				| Speed limit (50km/h) 							|
| 0  					| Speed limit (20km/h)              			|
| 0  	      			| Wild animals crossing  		         		|
| 0 				    | Speed limit (80km/h)						   	|

For the Fifth image ... 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Keep left   								    | 
| 0     				| Ahead only         							|
| 0  					| Turn right ahead'             				|
| 0  	      			| Go straight or left  		         			|
| 0 				    | Beware of ice/snow'						    |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


