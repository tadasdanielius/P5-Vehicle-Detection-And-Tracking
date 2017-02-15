# TODO: Instead of hardcoding make it possible to pass params whilst constructing object
class FeatureExtractionSettings:
    """ Store configuration parameters for feature extraction functions """
    def __init__(self):
        self.pix_per_cell = 8
        self.orient = 9
        self.window = 64
        self.cell_per_block = 2
        self.cells_per_step = 2
        self.hog_channel = 4  #4 - means all
        self.use_hog = True
        self.use_spatial = True
        self.use_hist = True
        self.spatial_size = (32, 32)
        self.hist_bins = 32
        self.scale = 1.0
        self.color_space = 'YCrCb'
        self.yregion = (400, 650)
        self.xregion = None


