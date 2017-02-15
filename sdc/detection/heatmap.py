import numpy as np
from scipy.ndimage.measurements import label
import cv2


class DynamicHeatmap:
    def __init__(self, shape, frames=10):
        self.frames = frames
        self.shape = shape
        self.heat = np.zeros((shape[0], shape[1])) #.astype(np.float)
        self.heatmaps = np.zeros((frames, shape[0], shape[1]))
        self.prev_thres = np.zeros_like(self.heat)

    def add_heat(self, boxlist):
        self.heat = np.zeros_like(self.heat)
        # Iterate through list of bboxes
        for box in boxlist:
            self.heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Keep list of latest heatmaps
        frames = self.frames - 1
        # Push maps back
        self.heatmaps[0: self.heatmaps.shape[0]-1, ] = self.heatmaps[1, ]
        # Add heatmap to the last position
        self.heatmaps[frames, ] = self.heat

        return self.mean_heatmap()

    def mean_heatmap(self):
        # Return mean of collected heatmaps
        avg_heat = self.heatmaps.mean(axis=0)
        return np.clip(avg_heat, 0, 255)

    def next_frame(self, sub, prop):
        self.heat -= sub
        self.heat -= (self.heat*prop).astype(np.int8)
        self.heat = np.clip(self.heat, 0, 15)
        return self.heat

    def gen_mask(self, mask, threshold):
        mask[mask <= threshold] = 0
        mask[mask != 0] = 1
        return mask

    def apply_threshold(self, threshold=2):
        result = self.heatmaps[0]
        for hm in self.heatmaps:
            mask = self.gen_mask(hm, threshold)
            result = cv2.bitwise_and(result, mask)
        return result

    def expand_rect(self, box, expand):
        """ Function to expand borders of the image by given number of pixels """
        point_1, point_2 = box
        x1, y1 = point_1
        x2, y2 = point_2

        # Choose max for x1 to avoid values less than 0
        x1 = max(0, x1-expand[0])
        # Choose min for x2 to avoid values larger than image x2
        x2 = min(x2+expand[0], self.shape[1])

        # Do the same for y axis
        y1 = max(0, y1-expand[1])
        y2 = min(y2+expand[1], self.shape[0])

        return ((x1, y1), (x2, y2))

    def find_heat_rectangles(self, threshold=2, expand=(10, 10), reduce=(-10, -10), conn=5):
        """ Find rectange of the labels in heatmap """
        # Apply threshold
        th = self.apply_threshold(threshold)
        labels = label(th) #self.gen_mask(self.heat, 0))


        # Find separate heat points and label them
        boxes = []
        for idx in range(labels[1]):
            obj_id = idx + 1
            # Pick only single object
            objects = np.copy(labels[0])
            objects[objects != obj_id] = 0
            # Apply mask
            i, j = np.where(objects)
            indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                                  np.arange(min(j), max(j) + 1),
                                  indexing='ij')
            # Find min and max index
            indices = np.array(indices)
            y1 = indices[0, :, :, ].min()
            y2 = indices[0, :, :, ].max()
            x1 = indices[1, :, :, ].min()
            x2 = indices[1, :, :, ].max()

            box = ((x1, y1), (x2, y2))
            # Expand borders of the object
            box = self.expand_rect(box, expand)
            cv2.rectangle(th, box[0], box[1], 1, -1)

        # TODO: need proper refactoring this part of the code should go into separate function
        labels = label(th)
        for idx in range(labels[1]):
            obj_id = idx + 1
            # Pick only single object
            objects = np.copy(labels[0])
            objects[objects != obj_id] = 0
            # Apply mask
            i, j = np.where(objects)
            indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                                  np.arange(min(j), max(j) + 1),
                                  indexing='ij')
            # Find min and max index
            indices = np.array(indices)
            y1 = indices[0, :, :, ].min()
            y2 = indices[0, :, :, ].max()
            x1 = indices[1, :, :, ].min()
            x2 = indices[1, :, :, ].max()

            box = ((x1, y1), (x2, y2))
            box = self.expand_rect(box, reduce)

            boxes.append(box)

        return boxes, th, labels
