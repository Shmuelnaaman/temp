# **Traffic Sign Recognition** 

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
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


Here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

Images are loaded and features and labels were set. These are 32x32x3 images which are color, not grayscale. Then the images are visualized from dataset. Then we got the histogram distribution of the trainning set, validation set and test set.

I used the pandas and numpy libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of test set is 12630 sample
* The size of traffic sign is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the distribution of training data is.

![output][./download]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, (pixel - 128)/ 128 is the way I used to normalize the data. I had used 5 images downloaded from web to know how well the model works. I shuffled the data to make sure that the order in which the data comes does not matters to CNN.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1)Input         		| 32x32x3 RGB image   							| 
| 2)Convolution       	| 1x1 stride, valid padding, outputs 28x28x6 	|
| 3)RELU				|output of 2 is activated   					|
| 4)Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6	|
| 5)Convolution 3x3	    |1x1 stride,valid padding,Output-10x10x16		|
| 6)RELU		        | output of 5 is activated						|
| 7)Max pooling			|2x2 stride, valid padding, outputs 5x5x16		|
| 8)Flatten input   	|												|
| 9)Fully connected		|Input = 400. Output = 120  					|
| 10)RELU				|output of 9 is activated						|
| 11)Fully connected	|Input = 120. Output = 84       				|
| 12)RELU				|output of 11 is activated						|
| 11)Fully connected	|Input = 84. Output = 43        				|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an EPOCHS = 50, BATCH_SIZE = 128. Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer mu = 0 and sigma = 0.1. The learning rate = 0.001. We use adam optimizer to minimise the loss function similarly to what stochastic gradient descent does. We ceate a training pipeline that uses the model to classify data. Then we evaluate how well the loss and accuracy of the model for a given dataset. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 94.9%
* test set accuracy of 94.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
   First I tried the predefined LeNet Architecture which is learned from the previous sessions(channels were updated to 3). It was running    well but with an accuracy of 90%. 
* What were some problems with the initial architecture?
   It was running well but with an accuracy of 90%.
* How was the architecture adjusted and why was it adjusted?
    I tried dropout with keep_prob after convolution layer 1,2 and first fully connected layer. Then also adjusted the parametres(epochs       from 10 to 50,learning rate from 0.005 to 0.001....) . Then the accuracy changed to 94%.
* What are some of the important design choices and why were they chosen? 
    Dropout is a very good regularization method. Here the model learns a redundant representation for everything to make sure that some of the informations remain.This makes model more arobust and prevents over fitting. This is agood way to improve performnce.


### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][./images/aheadonly35.jpg] ![alt text][./images/bicycle29.jpg] ![alt text][./images/pedestrian22.jpg] 
![alt text][./images/noentry17.jpg] ![alt text][./images/wildanimals31.jpg]


The test set accuracy is 40%. The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of . But sometimes the test set is giving 60% accuracy

Image 0 probabilities:  3562.85058594  3389.4387207    407.12240601   277.73754883  -131.81756592] 
 and predicted classes: 25 31 10 23 20
Image 1 probabilities:  5047.83984375  3421.85864258  2344.47631836  1036.56469727   366.73782349 
 and predicted classes: 11 26 27 12 20]
Image 2 probabilities:  3876.63916016  1753.13037109   979.190979     886.00085449   516.81860352
 and predicted classes: 35 12 25 20  3
Image 3 probabilities:  2587.58935547    99.08230591    10.31815624  -237.60853577  -344.88641357
 and predicted classes: 34 33 38 35 17
Image 4 probabilities:  2499.07592773  2076.40942383   252.05569458   112.24391174    43.91285324 
 and predicted classes: 29 28 20 25 23




