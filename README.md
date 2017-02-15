The  final video can be found [Here](https://youtu.be/XlZunXxhPKE)

Writeout notebook can be found [Here](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/writeout.ipynb)

# Vehicle tracking

This project goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4).

## The project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Perform color transform and append binned color features
* Addad histograms of color
* Combine all feature vectors along with HOG feature vector.
* Normalize the all feature vectors
* Estimate a bounding box for vehicles detected.
* Implement a sliding-window technique and use on the trained classifier to search for vehicles in images.

Initially experiments were performed on the test.video.mp4 video stream and later implement on full project_video.mp4.

### Exploring the training data

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples which I used to train my classifier. These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/).

Training set was split into 80% training data and 20% testing data:

<pre>
Total images: 17760
Total number of vehicles: 8792
Total number of non vechicles: 8968

Training set: 14208
Testing set: 3552
</pre>

![Training images](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/images/image1.png)

There are some examples provided in test_images folder for testing the algorithm and processed results saved into output folder.
![Test images](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/images/image2.png)

## Feature extraction

### Histogram of Oriented Gradients (HOG)

The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image. This method is similar to that of edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts, but differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy.
Here is some examples on test data
![HOG](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/images/image3.png)

### Parameters for HOG

Parameters were chosen based on trial and error. I have tried different approaches and considering speed and accuracy I have chosen the following settings:

<pre>
Orientation: 9
Cells per block: 2
Pixel per cell: 8
</pre>

Increasing or decreasing orientation from 7 to 12 does not give any benefits. Increasing number of pixels per cell leads to poorer performance and less accuracy. So this combination of parameters worked best

### Colour histogram features

The other source of features I used is histograms of pixel intensity (color histograms) as features. I used all three channels to generate histogram and vectorize everything. With some experimentation I found that converting to **UCrCb** works best with least false positives. **LAB** and **LUV** colour spaces also did a great job, however it gave more false positive. I also tried **HSV**, **YUV** and **HLS** but those spaces gave too much false positives

Here is an example of different colour spaces
![Color Spaces](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/images/image4.png)

### Spatial binning

Finally the last source of features comes from raw pixel. It is quite useful to include in my feature vector in searching for cars. First, I resized image to (32, 32) to make it smaller vector but enough to retain the information. Since classifier did a good job I decided not to invest more time in tweaking parameters but it is something worth trying in the future like changing colour spaces using only one channel making even smaller image etc.

## Training classifier

All features combined gave me 8460 feature vector for each training sample. The feature vector variance as expected is high because features are combined from different domain, thus feature normalization as in any machine learning application is needed for good results. I used Python's sklearn package which provides [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) method to accomplish this task by standardizing features by removing the mean and scaling to unit variance. My choice of classifier was [Linear Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine).
Training process was quite fast but feature extraction took bit longer. In total it took around 77 secconds to extract features, apply scaler and train SVM

<pre>
Accuracy: 0.9932432432432432
Took time to train: 77 seconds
</pre>

The full ipython notebook can be found in [training.ipynb](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/training.ipynb)

## Detecting vehicles

First, the images are large enough and really we don't need such resolution, therefore I decided to resize it in half from (720, 1280) to (360, 640). Also, we don't need to search for vehicles over the whole image. Cropping out the area of interest from the whole image will speed up detection and reduce the number of false positives. I choose to use only x axis from 330 to 640 and y axis from 200 to 330.

![Region](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/images/image5.png)

The area is rather small and X axis should be extended, but for this case it works well. And my goal is to balance between quality and performance

### Sliding window

To detect vehicles in the image I implemented sliding window technique to search accross the area of interest. The window size I chose is 64 pixels. The number of steps is calculated:

<pre>
imshape = converted_img.shape
nxblocks = (imshape[1] // pix_per_cell) - 1
nyblocks = (imshape[0] // pix_per_cell) - 1

nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
nysteps = (nyblocks - nblocks_per_window) // cells_per_step
</pre>
Here **pix_per_cell = 8, cells_per_step = 2**. So it makes **60** windows in total.

The code is defined in [sdc.detection.object_detection.py](object_detection.py) starting from line 28
The area is rather small and X axis should be extended, but for this case it works well. And my goal is to balance between quality and performance

### Sliding window

To detect vehicles in the image I implemented sliding window technique to search accross the area of interest. The window size I chose is 64 pixels. The number of steps is calculated:

<pre>
imshape = converted_img.shape
nxblocks = (imshape[1] // pix_per_cell) - 1
nyblocks = (imshape[0] // pix_per_cell) - 1

nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
nysteps = (nyblocks - nblocks_per_window) // cells_per_step
</pre>
Here **pix_per_cell = 8, cells_per_step = 2**. So it makes **60** windows in total.

The code is defined in [sdc.detection.object_detection.py](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/sdc/detection/object_detection.py) starting from line 28

![Sliding window](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/images/image6.png)

However that isn't good enough so I also use scaling factor, which basically enlarges HOG transformed image making searching space larger.
In my case the image is scaled by **0.7**. So, that makes image dimensions from **320x130** to **457x185** and number of windows increases to **168**

![Sliding window](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/images/image7.png)

### Heatmap

Some windows are overlapping and in some areas we get false positive. In order to combine overlapping detections and remove false positives I build a heatmap. To make a heat-map, I simply add "heat" (+=1) for all pixels within windows where a positive detection is reported by classifier. The functionality of the heatmap is implemented in file [sdc.detection.heatmap.py](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/sdc/detection/heatmap.py) 

The individual heat-maps for the above images look like this:
![Heatmap](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/images/image8.png)

### Labelling objects

Although I have experimented with different approaches and different thresholding to convert heatmap to binary mask. In the end I decided to take different approach. The heatmap isn't really required in my approach because I am using every point which is larger than 0. That means I don't need to setup any threshold. So, basically every point is already "included" when labelling.
Using binary image I then label each detected object with [label()](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html) method. Which helps me to get the boundaries of the object.
Here is an example of the labelled objects
![labels](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/images/image9.png)

### Getting rid of false positive

I have tried different approaches on how to remove false positive like averaging heatmap over number of frames which should remove accidently appearing on a single frame, however it requires tunning threshold parameter. Instead of averaged heatmap I decided to apply **bitwise_and** operation between previous frame and current frame, and that works well.

### Merging boxes

There is one more problem left to sort out. If vehicle is close enough it takes a lot of spaces the classifier identifies "correctly" different places of vehicle, however during labelling step I get 2 or even more labelled objects but it should be treated as one. Here is illustrative example what could happen:

![box_merge](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/images/image10.png)

Clearly this belongs to the same vehicle, but labelled as separate objects. To solve this problem I use expand and relabel techinque. It works in the following way, on the binary mask I redraw rectangles with slightly expanded borders by 10 pixels. So the result might look like this:

![box_merge](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/images/image11.png)
And relabel objects again. Those boxes are now detected as single objects. That solves the problem

### Pipeline

And finally I put everything together in pipeline. Now, the challenge is to find the balance between performance and quality. After some experimentation I choose to use only 10th frame and for the rest  just draw the previously detected borders. This works well with getting rid of false positive, since I use **bitwise_and** operation every 10th frame and incorrectly detected areas will likely disappear and ofcourse skipping 9 frames boosts the performance. The **project_video** was processed in than **38 seconds** that is even less than actuall video which is **51 second** !

pipeline notebook file can be found [here](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/pipeline.ipynb)

## Weakness of the algorithm

The strength of the implementation I believe is fast performance it processes video much faster than the length of actual video. However there are some issues with this:

* The parameters are fine-tuned for the project video but I see there could problems with other videos.
* Monitored area is very small compared with actual image. 
* Ignoring 9 frames can be a problem if vehicles are moving fast or quickly stopping. 
* The captured box is jumping and actual size of the car is not captured well.
* There is no proper mechanism for tracking vehicle between frames.
* The image is resized by half, so vehicles which are further away will not be detected

Instead of SVM I have tried to classify same features using neural network implementation instead. The results were not so impressive as I expected, but since it was not a requirement I decided not to invest much time in tunning the architecture of the neural network and tweaking the hyper parameters. Also, the performance of predictions were much slower than SVM, but I guess this is a problem with implementation in Keras. 

Final results can be found [here](https://youtu.be/XlZunXxhPKE)
Writeout notebook can be found [here](https://github.com/tadasdanielius/P5-Vehicle-Detection-And-Tracking/blob/master/writeout.ipynb)
