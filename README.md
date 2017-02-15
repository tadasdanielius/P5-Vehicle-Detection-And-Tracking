The  final video can be found [Here](https://youtu.be/XlZunXxhPKE)

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

There are some examples provided in test_images folder for testing the algorithm and processed results saved into output folder.


