import cv2
import numpy as np
from skimage.feature import hog
from sdc.image_processing.helpers import equalize_color_hist
from sdc.image_processing.helpers import convert_to_color_space
from sdc.image_processing.feature_extraction_settings import FeatureExtractionSettings


# Function to return HOG features and visualization
# http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """ Extracts hog features from the image """
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    """ function to compute binned color features """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    """ Computes color histogram features for each channel """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features(images, settings):
    """ Extract features from the list of image files."""
    # Create a list to append feature vectors to
    features = []

    for file in images:
        # Read the file
        image = cv2.imread(file)
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = equalize_color_hist(image)
        # Get feature list for a single file
        image_features = single_img_features(image, settings)
        # Append to feature collection
        features.append(image_features)

    return features


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, settings):
    settings = settings if settings is not None else FeatureExtractionSettings()

    # 1) Define an empty list to receive features
    img_features = []
    feature_image = np.copy(img)
    # 2) Apply color conversion if other than 'RGB'
    feature_image = convert_to_color_space(feature_image, settings.color_space)
    # 3) Compute spatial features if flag is set
    # 7) Compute HOG features if flag is set
    if settings.use_hog:
        if settings.hog_channel == 4:
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     settings.orient, settings.pix_per_cell, settings.cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, settings.hog_channel], settings.orient,
                                            settings.pix_per_cell, settings.cell_per_block,
                                            vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)
    if settings.use_spatial:
        spatial_features = bin_spatial(feature_image, size=settings.spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if settings.use_hist:
        hist_features = color_hist(feature_image, nbins=settings.hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)

    # 9) Return concatenated array of features
    return np.hstack(img_features)
