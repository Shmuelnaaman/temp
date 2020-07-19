# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

### Data Set Summary & Exploration

#### 1. German Traffic Sign Dataset Details:

* Training, Validation and Test data is loaded into corresponding variables with respective labels using pickle.
    * Exploration of the datasets indicated :
        * Number of training examples = 34799
        * Number of validation examples = 4410
        * Number of testing examples = 12630
        * Image data shape = (32, 32, 3)
        * Number of classes = 43

* Training, Validation, Test Split percentages used= 67.13%, 8.51%, 24.36%

#### 2. Exploratory visualization of the dataset.
We first analyezed the distribution of Training and Test dataset for different classes to understand any limitations of the datsets
![Distribution of dataset across classes](./Images_for_writeup/Train_Test.png)
It is observed that few classes have high percentage of images Vs few have very few images.

Sample data was explored visually by randomly checking sample image for each class:
![Sample_Image_Data](./Images_for_writeup/SampleDataset.png)


### Design and Test a Model Architecture

#### 1. Pre-processing of Images:

Before feeding an image to a Model, preprocessing of the image is performed using following piepline:
**Preprocessing pipeline :: Image -->Grayscale --> Equalize**

1.) Convert the images to grayscale: Useful for two main reasons-
    *a.) When distinguishing between traffic signs, Color is not a very significant feature to look for. The lighting in our image also varies. The features of the traffic signs that really matter are the edges the curves the shape and side of it.
    b.) Using grayscale we reduce the depth of our image from 3 to 1. Means network now requires fewer parameters as our input data. Resulting in much more efficient model processing and will require less computing power to classify.*

![Gray Scale Conversion](./Images_for_writeup/Gray_Conversion.png)

2.) Histogram Equalization:: Contrast of each image is adjusted by means of histogram equalization. This is to mitigate the numerous situation in which the image contrast is really poor.

![Histogram Equalized Imaged](./Images_for_writeup/Hist_equalize.png)

#### 2. Data Augumentation:
ImageDataGenerator available in the Keras library used to augument the data.
Data augmentation is perfromed online, during the training by performing following actions: 
    a.) Randomly Rotated
    b.) Zoomed
    c.) Shifted: Horizontally and vertically
    d.) Shear
    
This was done to create some variety in the data while not completely eliminating the original feature content. 

Samples of process of augmentation is visible below::


#### 3. Model Architecture:

My final impleted model after mutilple experiments and tunings is represented as:
![Model_Visualization](./Images_for_writeup/model.png)

The model consists for following layers:

Layer (type)                 Output Shape              Param #   
--------------------------------------------------------------
conv2d_1 (Conv2D)            (None, 28, 28, 60)        1560      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 60)        90060     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 60)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 10, 10, 30)        16230     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 30)          8130      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 30)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 480)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 500)               240500    
_________________________________________________________________
dropout_1 (Dropout)          (None, 500)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 43)                21543     
_________________________________________________________________
Total params: 378,023
Trainable params: 378,023
Non-trainable params: 0
_________________________________________________________________
None

**As can be seen the model is fairly simple having 4 convolutional layers wherein 2 convulutional layers are followed by max pooling layer. 
Dropout layer is used to minimize the imapact of overfitting.**

*'relu' activation function is used for all the layers except for the last Dense layer (Fully Connected layer- softmax activation') to help classify input into one of the 43 classes.*

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model used an Adam optimizer, with learning rate of '0.001'.
loss='categorical_crossentropy'
metrics='accuracy'

Employed Batch generator (batch_size=128) as it helps to create augmented images on the fly rather than augmenting all your images at one time and storing them using valuable memory space.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9817 
* validation set accuracy of 0.9900
* test set accuracy of 0.977

My model is derieved from leNet model (https://en.wikipedia.org/wiki/LeNet#:~:text=LeNet%20is%20a%20convolutional%20neural,a%20simple%20convolutional%20neural%20network)

To begin with, I used leNet model for classification.
When Trained the network on this model, I observed, accuracy is not as high and network seems to be overfitting the training data.

So fine tuned the model after few iterations. Reduced the learning rate, coupled existing convolution layers with additional convolution layers and increased number of filters.

![Loss](./Images_for_writeup/Loss_plots.png)

![Accuracy](./Images_for_writeup/acc_plot.png)


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![External_Images]('./Images_for_writeup/ExternalImages.png')


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Truth Labels for the Images:
* 	Image	SignNum
* Bumpy_road.jpeg	22
* Construction.jpeg	25
* Yield.jpeg	13
* Slippery_road.jpeg	23
* Speed_limit50.jpeg	2

Corresponding Truth and Predicted Results:
* (22, array([22]))
* (23, array([23]))
* (13, array([13]))
* (25, array([25]))
* (2, array([2]))


The model was able to correctly guess all of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

To determine top five soft max probabilities for each image, I used model.predict function.
Model.predict outputs the probabilities for the Input image in all 43 classes.

I then selected the top 5 probability predictions for each of the 5 external images:

* Image 1: Bumpy_road.jpeg
[[  9.97937977e-01]
 [  2.02207849e-03]
 [  1.23137061e-05]
 [  8.34628099e-06]
 [  7.42188467e-06]] 
    * Corresponding Classes for these probabilites:
    [22, 9, 40, 25, 17]

* Image 2: Slippery Road
[[  1.00000000e+00]
 [  1.22007622e-08]
 [  1.00873905e-08]
 [  7.74345865e-11]
 [  1.58160013e-11]] 
    * Corresponding Classes for these probabilites:
     [23, 29, 19, 30, 9]

* Image 3: Yield
[[  1.00000000e+00]
 [  1.17654722e-08]
 [  6.29163655e-09]
 [  2.21135399e-09]
 [  1.56579250e-09]] 
    * Corresponding Classes for these probabilites:
     [13, 12, 25, 3, 5]
 
* Image 4: Construction
[[  9.99997497e-01]
 [  2.50003723e-06]
 [  2.07870943e-09]
 [  6.77481349e-10]
 [  2.38146974e-10]]
    * Corresponding Classes for these probabilites:
    [25, 22, 11, 19, 10]

* Image 5: Speed_limit50
[[  1.00000000e+00]
 [  1.78232741e-13]
 [  1.17783859e-15]
 [  5.43006287e-17]
 [  1.28891000e-20]] 
    * Corresponding Classes for these probabilites: 
     [2, 5, 3, 1, 0]

