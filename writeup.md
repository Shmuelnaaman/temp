# **Traffic Sign Recognition | Writeup | Mohit Arvind Khakharia** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./custom_pics/1.png "Traffic Sign 1"
[image2]: ./custom_pics/2.png "Traffic Sign 2"
[image3]: ./custom_pics/3.png "Traffic Sign 3"
[image4]: ./custom_pics/4.png "Traffic Sign 4"
[image5]: ./custom_pics/5.png "Traffic Sign 5"
[image6]: ./custom_pics/5.png "Traffic Sign 5"
[image7]: ./images_for_writeup/custom_image_processing.png "Custom_image_processing"
[image8]: ./images_for_writeup/custom_images.png "Custom_images"
[image9]: ./images_for_writeup/dataset_augmented.png "Dataset_augmented"
[image10]: ./images_for_writeup/frequency_analysis.png "Frequency_analysis"
[image11]: ./images_for_writeup/grayscale_conversion.png "Grayscale_conversion"
[image12]: ./images_for_writeup/Lenet.png "Lenet"
[image13]: ./images_for_writeup/optimizers.png "Optimizers"

---
### Project Code

### Data Set Summary & Exploration

### Before Augmentation
Size and shape of the training set
Size - 34799 Samples
Shape - 34799 x 32 x 32 x 3

 ```
* Size and shape of the Vlidation set
```
Size - 4410 Samples
Shape - 4410 x 32 x 32 x 3
```
* Size and shape of the test set
```
Size - 12630 Samples
Shape - 126302 x 32 x 3
```

* Shape of a Traffic Sign image
``` 
32 x 32
```
* Unique Classes/Labels in the data set
```
43
```

### Problem Aspects considered

- How to avoid over or underfitting?
- Are techniques like normalization, rgb to grayscale, shuffling needed
- Number of examples per label (some have more than others).
- Should we generate fake data / augmentation.
The decisions taken are described below.

## Data Augmentation

As mentioned in the lecutre, it is always a good practice to augment the data before training inorder to achieve translationaly, rotational and brightness invariance.

### Trasnformations Applied
- Rotation
- Shear
- Zoom
- Translation
### Dataset size increase
 - ```Training Data new size = 69598 | Approximately 2x```
 - ```Validation Data new size = 17640 | Approximately 1.5x```
 
### Visualization of the Augmented dataset.

![alt text][image9]

### Statistical Analysis of the Class frequency in the Augmented data
- Tells us if we have enough samples of all the classes
- Tells us if our training data is not biased
- Tells us that if we cover all the classes in the Validation set
- Tells us if we test all the classes in the Test set

![alt text][image10]

## Model Architecture
- The architecture implemented was a modified version of LeNet-5 shown in the  [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81).

### Preprocessing the augmented dataset

- Normalized the data has mean zero and equal variance by using the following formula - `(pixel - 128)/ 128`
- Converted to grayscale by averaging the channels.
- Shuffled the data for preventing a biased learning curve when using batch based Gradient algorithms as they assume the batch is an approximation of the entire dataset.

```
Training Data Shape - Before Preprocessing
(69598, 32, 32, 3)
Training Data Shape - After Preprocessing
(69598, 32, 32, 1)
Validation Data Shape - Before Preprocessing
(17640, 32, 32, 3)
Validation Data Shape - After Preprocessing
(17640, 32, 32, 1)
Testing Data Shape - Before Preprocessing
(12630, 32, 32, 1)
Testing Data Shape - After Preprocessing
(12630, 32, 32, 3)
```

![alt text][image11]

### REASON
- Converting to grayscale -It helped to reduce training time and makes detection color agnostic.
- Normalizing the data to the range (-1,1) - As mentioned in the classes, a wider distribution in the data would make it more difficult to train using a singlar learning rate. Also, the math becomes difficult at extremely large [or] extremenly small numbers.

#### Original LeNet Model Architecture
![alt text][image12]

The architecture I used was a modified version of LeNet as it already did a preety good job of classifying images and need a few tweaks to get exceptional performance.

Custom Modified Architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												| outputs 28x28x6
| Max pooling	      	| 1 2 2 1 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | outputs 10x10x16     									|
| RELU					|												| outputs 10x10x16  
| Max pooling	      	| 1 2 2 1 stride,  outputs 5x5x16  				|
| Flattening	      	| output 400  				|
| Fully connected		| output 200        									|
| Fully connected		| output 100        									|
| Softmax				| Result output 43        									|


## Training the model
### Hyper Parameters and experimentation
- EPOCHs = 100 - Tried with other epochs but it did not imporve the accuracy a lot. Higher values caused overfitting and small values caused under fitting.
- LEARNING RATE = 0.001 - After experimenting it was realized that 0.001 is a really good learning rate for 100 epochs.
- MU = 0
- SIGMA = 0.1 - Gives a really good normal distributon for choosing the random weights.
- KERNEL SIZE = 5 - Experimentation proved that it was a good size for an image of size 32 x 32
- BATCH SIZE = 128 - Tried with higher batch sizes but only resulted in more memory and time without any noticable increase in accuracy.
- KEEP PROBABILITY = 1.0 - This method is used more of a replacement to max-pooling now days but since I already had  max-pooling doing the job for me, I did not face the need to change the "Keep Probabilty" to lower numbers.

### Optimizer
*AdamOptimizer*
- This works really well because this method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.
- Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods.

![alt text][image13]

### Fancy Stuff experimented
- Tried to implement Hypertuning. An approach to autmatically find the best set of parameters.
- Played around with Tensor board and TensorFlow summaries for visualization.
- Played around with a capsule network solution for image classification.
- Classification using pretrained inception model by retraining the last few layers.

#### FINAL ACCURACY
| Dataset        	|     Accuracy        					| 
|:---------------------:|:---------------------------------------------:| 
| Validation         			| 94.2%   									| 
| Test    				|93.1% 										|


### Model on Custom Images
- The model was experimented on 5 custom images.
- The same preprocessing steps of Normalization and Grayscaling were applied.
- An additional step of resizing was applied.
```
Custom Images and Shapes
img_i: 1.png
image.shape  (32, 32, 4)

img_i: 2.png
image.shape  (32, 32, 4)

img_i: 3.png
image.shape  (32, 32, 4)

img_i: 4.png
image.shape  (32, 32, 4)

img_i: 5.png
image.shape  (32, 32, 4)

Gray scaled image array shape - (5, 32, 32, 1)
```
![alt text][image8]

## Prediction for custom images
- The tf.nn.top_k was used fpr predicting the top 3 guesses for each image.
- Please find the proability of each guess mentioned on the image label

![alt text][image7]

### RESULTS FOR CUSTOM IMAGES
- The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.
```
Image -  0
Guess 1 Probability-  0.976936
Guess 2 Probability-  0.0121746
Guess 3 Probability-  0.0108897
Image -  1
Guess 1 Probability-  1.0
Guess 2 Probability-  7.67467e-13
Guess 3 Probability-  4.498e-26
Image -  2
Guess 1 Probability-  1.0
Guess 2 Probability-  1.82335e-18
Guess 3 Probability-  2.00922e-22
Image -  3
Guess 1 Probability-  1.0
Guess 2 Probability-  0.0
Guess 3 Probability-  0.0
Image -  4
Guess 1 Probability-  1.0
Guess 2 Probability-  2.11481e-07
Guess 3 Probability-  1.78538e-11

```

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 98         			| General Caution   									| 
| 99.9     				| Speed Limit 30 										|
| 99.8					| Keep Right											|
| 99.9	      			| Yield					 				|
| 99.9				    | Stop (Misclassified)      							|

### Future Improvements
- The model needs additional training data for a few classes so that the learning is not biased.
- The model failed to recognize in a few cases were the images were zoomed and thus we need to add zoom augmentated data.
- Need to integrate TensorFlow summaries for detailed debugging and analysis.

