#**Traffic Sign Recognition**

Here is a link to my [project code](https://github.com/zxy1256/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set.

![Example Images][traffic_signs_and_name.png]

There are many images that are dark. Normalization can make the brightness more
consistent. The following image is a histogram of the training, validation and
test set. They have similar distribution of signs.

![Sign Histogram][histogram.png]

###Design and Test a Model Architecture

As a first step, I normalized the image data because there are many images that
are very dark in the training set.

As a last step, I decided to convert the images to grayscale because for given
network architecture, grayscale can help the network learn more useful lower
level features.

Here is an example of a traffic sign image before and after normalization and
grayscaling.

![Original, Normalized and Grayscale][preprocess_demo.png]

I decided to generate additional data because that would help prevent overfitting.

To add more data to the the data set, I generated a rotated image for every
training image. The rotation degree is a random value between -40 and 40 degree.

Here is an example of an original image and an augmented image:

![Original and rotated][rotation_demo.png]

The augmented data set contains twice the number of images than the original
data set.

My final model consisted of the following layers:

| Layer           |     Description	        					            |
|:---------------:|:---------------------------------------------:|
| Input         	| 32x32x1 grayscale image						            |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6 	  |
| RELU					  |												                        |
| Max pooling	    | 2x2 stride,  valid padding, outputs 14x14x6 	|
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x48 	|
| RELU					  |												                        |
| Max pooling	    | 2x2 stride,  valid padding, outputs 5x5x48		|
| Flatten   	    | outputs 1200                              		|
| Fully connected	| inputs 1200, outputs 120                      |
| Fully connected	| inputs 120, outputs 84       									|
| Dropout	        |                             									|
| Fully connected	| inputs 84, outputs 43        									|
| Softmax				  |                                               |


To train the model, I used an AdamOptimizer with learning rate 0.001, batch size
128 and epochs 40.

My final model results were:
* training set accuracy of 98.5%
* validation set accuracy of 93%
* test set accuracy of 91.5%

Both my first and final models are based on LeNet. I think it can be suitable
for the current problem because LeNet has been used successfully on digit
recognition problems.

My first architecture uses RGB image rather than grayscale image. I chose to
use RGB image because I thought color is a good indicator of traffic sign. For
example, "stop" sign is red, and "keep right" sign is blue. The training accuracy
is not good enough with RGB images. Switching to grayscale improves training
accuracy significantly.

Another difference between first and last architecture is the second convolution
layer output depth is 16 in first architecture rather than 48. This is the value
copied from LeNet. I think increase the depth would help the network capture more
features and help improve accuracy, because the number of classes to classify is
more than 10 digit.

The last difference is the first architecture does not have a dropout layer.
Without the dropout layer, I got over 90% validation accuracy, but test accuracy
is not good.

###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![stop][stop.png] ![70 km per hour][70kmph.png] ![30 km per hour][30kmph.png]  ![curve][curve.png] ![traffic signs][traffic_signs.png]

Here are the results of the prediction:

| Image			                   | Prediction	        					|
|:----------------------------:|:----------------------------:|
| Stop	      		             | Stop					 				        |
| 70 km per hour               | 70 km per hour 						  |
| 30 km per hour      	       | 30 km per hour   						|
| dangerous curve to the right | dangerous curve to the right |
| Traffic signs		             | Traffic signs      					|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of
91.5%.

The following image shows the top 5 softmax probabilities for each image

![top5 probabilities][top5_prob.png]
