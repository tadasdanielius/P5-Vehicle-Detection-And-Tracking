import sdc.image_processing.helpers as iph
from sdc.detection.helper_functions import get_hog_features
from sdc.detection.helper_functions import bin_spatial
from sdc.detection.helper_functions import color_hist

import cv2
import numpy as np


class ObjectDetection:
    def __init__(self, classifier):
        self.classifier = classifier
        self.settings = self.classifier.settings

    def region_boundaries(self, img):
        yregion = self.settings.yregion if self.settings.yregion is not None else (0, img.shape[0])
        xregion = self.settings.xregion if self.settings.xregion is not None else (0, img.shape[1])
        return yregion, xregion

    def region(self, img):
        yregion, xregion = self.region_boundaries(img)
        ystart, ystop = yregion
        xstart, xstop = xregion
        return img[ystart:ystop, xstart:xstop, :], yregion, xregion

    def prepare_image(self, img):
        """ Scale down convert to color spaces in which model was trained"""
        image_to_search, _, _ = self.region(img)
        scale = self.settings.scale if self.settings.scale is not None else 1
        color_space = self.settings.color_space if self.settings.color_space is not None else 'YCrCb'
        image_to_search = iph.convert_to_color_space(image_to_search, color_space)
        if scale != 1:
            imshape = image_to_search.shape
            resize_to = (np.int(imshape[1]/scale)), (np.int(imshape[0]/scale))
            resized = cv2.resize(image_to_search, resize_to)
            return resized
        else:
            return image_to_search

    def prepare_features(self, img):
        """ This function will extract HOG features from the given image """
        # Define blocks and steps, how many blocks per cell
        pix_per_cell = self.settings.pix_per_cell if self.settings.pix_per_cell is not None else 8
        orient = self.settings.orient if self.settings.orient is not None else 9
        window = self.settings.window if self.settings.window is not None else 64
        cell_per_block = self.settings.cell_per_block if self.settings.cell_per_block is not None else 2
        cells_per_step = self.settings.cells_per_step if self.settings.cells_per_step is not None else 2
        hog_channel = self.settings.hog_channel if self.settings.hog_channel is not None else 4
        use_hog = self.settings.use_hog if self.settings.use_hog is not None else True

        converted_img = self.prepare_image(img)

        imshape = converted_img.shape
        nxblocks = (imshape[1] // pix_per_cell) - 1
        nyblocks = (imshape[0] // pix_per_cell) - 1
        # How many features we are going to extract
        nblocks_per_window = (window // pix_per_cell) - 1
        # How many steps we make accross HOG array
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        hogx = (None, None, None)

        if use_hog:
            if hog_channel == 4:
                ch1 = converted_img[:, :, 0]
                ch2 = converted_img[:, :, 1]
                ch3 = converted_img[:, :, 2]
                hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
                hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
                hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
                hogx = (hog1, hog2, hog3)
            else:
                ch = converted_img[:, : hog_channel]
                hog_ch = get_hog_features(ch, orient, pix_per_cell, cell_per_block, feature_vec=False)
                hogx = (hog_ch, None, None)

        return converted_img, (nxsteps, nysteps), nblocks_per_window, hogx

    def find_objects(self, img, visualize_sliding_window=False):
        yregion, xregion = self.region_boundaries(img)
        ystart, ystop = yregion
        xstart, xstop = xregion
        converted_img, steps, nblocks_per_window, hogx = self.prepare_features(img)

        cells_per_step = self.settings.cells_per_step if self.settings.cells_per_step is not None else 2
        hog_channel = self.settings.hog_channel if self.settings.hog_channel is not None else 4
        use_hog = self.settings.use_hog if self.settings.use_hog is not None else True
        use_spatial = self.settings.use_spatial if self.settings.use_spatial is not None else True
        use_hist = self.settings.use_hist if self.settings.use_hist is not None else True
        window = self.settings.window if self.settings.window is not None else 64
        pix_per_cell = self.settings.pix_per_cell if self.settings.pix_per_cell is not None else 8
        spatial_size = self.settings.spatial_size if self.settings.spatial_size is not None else (32, 32)
        hist_bins = 32 if self.settings.hist_bins is None else self.settings.hist_bins
        scale = 1 if self.settings.scale is None else self.settings.scale

        nxsteps, nysteps = steps
        count = 0

        draw_img = np.copy(img)
        boxes = []
        for xb in range(nxsteps):
            for yb in range(nysteps):
                features = []
                count += 1
                ypos, xpos = yb*cells_per_step, xb*cells_per_step
                # Should we use HOG features
                if use_hog:
                    if hog_channel == 4:
                        hog1, hog2, hog3 = hogx
                        hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

                        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                    else:
                        hog1, _, _ = hogx
                        hog_features = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

                    features.append(hog_features)
                xleft, ytop = xpos * pix_per_cell, ypos * pix_per_cell

                subimg = cv2.resize(converted_img[ytop:ytop + window, xleft:xleft + window], (window, window))

                # Should we use spatial features
                if use_spatial:
                    spatial_features = bin_spatial(subimg, size=spatial_size)
                    features.append(spatial_features)

                # Should we use colour histogram features
                if use_hist:
                    hist_features = color_hist(subimg, nbins=hist_bins)
                    features.append(hist_features)

                all_features = np.hstack(features)
                prediction = self.classifier.predict(all_features)

                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                if visualize_sliding_window:
                    cv2.rectangle(draw_img,
                                  (xbox_left+xstart, ytop_draw+ystart),
                                  (xbox_left+win_draw+xstart,
                                   ytop_draw+win_draw+ystart), ((xb*yb) % 255, 255, 0), 1)

                if prediction == 1:
                    boxes.append((
                        (xbox_left + xstart, ytop_draw + ystart),
                        (xbox_left + win_draw + xstart, ytop_draw + win_draw + ystart)))
                    cv2.rectangle(draw_img,
                                  (xbox_left+xstart, ytop_draw+ystart),
                                  (xbox_left+win_draw+xstart,
                                   ytop_draw+win_draw+ystart), (255, 0, 0), 3)
        return draw_img, boxes
