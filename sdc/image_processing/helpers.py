import cv2
import numpy as np
import matplotlib.pyplot as plt


def equalize_color_hist(img):
    """ Equalizes colour image histogram """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return result


def convert_to_color_space(img, color_space='RGB'):
    """ Converts RGB image to given new color space"""
    new_image = np.copy(img)
    if color_space != 'RGB':
        convert_to_space = getattr(cv2, 'COLOR_RGB2{}'.format(color_space))
        new_image = cv2.cvtColor(img, convert_to_space)
    return new_image


def visualize(fig, rows, cols, imgs, titles, cmap='hot'):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap=cmap)
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])


