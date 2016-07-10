import os
from PIL import Image
import numpy
import math
from scipy.signal import medfilt

def get_image_channels(image: Image.Image):
    from itertools import count
    return dict(zip(image.getbands(), count()))

def get_channel_data(image: Image.Image, *channel_names: str, as_dict: bool = False):
    width, height = image.size
    channels = get_image_channels(image)
    channel_data = {}
    for channel_name in channel_names:
        channel_index = channels[channel_name]
        channel_data[channel_name] = numpy.array(image.getdata(channel_index)).reshape(height, width)
    if as_dict or len(channel_names) != 1:
        return dict(zip(channel_names, (channel_data[channel_name] for channel_name in channel_names)))
    else:
        return channel_data[channel_names[0]] 


def average_regions(ary, split_points):
    from numpy import asarray, empty, ndindex
    ary = asarray(ary)
    split_points = [asarray(points) for points in split_points]

    if len(split_points) != len(ary.shape):
        raise ValueError("`split_points` must contain a sequence of indices for each dimension of `ary`")
    dim = len(split_points)

    result_shape = tuple(len(points) for points in split_points)
    average = empty(shape=result_shape)

    for index in ndindex(result_shape):
        region = [slice(*split_points[d][index[d]]) for d in range(dim)]
        average[index] = ary[region].mean()

    return average

class ImageProcessor:
    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.width, self.height = image.size # type: int, int
        self.bbox_left, self.bbox_right = 0, self.width
        self.channels = get_image_channels(image)
        self.channel_data = {}

    def get_channel_data(self, *channel_names: str, as_dict: bool = False):
        return get_channel_data(self.image, *channel_names, as_dict=as_dict)

    def extract_stripe(self, top, bottom):
        cropped_width = self.bbox_right - self.bbox_left
        new_x = (self.width - cropped_width) // 2
        print('cropped_width: {} ({} -> {})'.format(cropped_width, self.bbox_left, self.bbox_right))
        board = Image.new('RGBA', (self.width, bottom - top), 'white')
        pasted = board.copy()
        pasted.paste(self.image.crop(( self.bbox_left, top, self.bbox_right - 1, bottom - 1 )), (new_x, 0))
        board = Image.alpha_composite(board, pasted)
        return board.convert('LA')

    def extract_stripes(self, stripes):
        return [self.extract_stripe(top, bottom) for (top, bottom) in stripes]

    def average_regions(self, region_size):
        from numpy import arange, column_stack
        white_bg = Image.new('RGBA', self.image.size, 'white')
        blended = Image.alpha_composite(white_bg, self.image.convert('RGBA')).convert('LA')

        im = 1 - get_channel_data(blended, 'L') / 255
        sx, sy = region_size, region_size
        grid_x = arange(0, self.width + sx, sx)
        grid_y = arange(0, self.height + sy, sy)

        regions = ( column_stack((grid_y[:-1], grid_y[1:])),
                    column_stack((grid_x[:-1], grid_x[1:])), )
        averages = average_regions(im, regions)
        averages = medfilt(averages, 5)
        return averages

    def find_bbox(self):
        from math import ceil
        from numpy import array, nonzero
        region_size = math.ceil(min(self.width, self.height) / 200)
        averages = self.average_regions(region_size)

        max_per_column = averages.max(axis=0)
        threshold = 0.03

        left, right = array(nonzero(max_per_column > threshold)).flatten()[[1, -1]] * region_size
        right += region_size # set right boundary to end of region

        self.bbox_left = left
        self.bbox_right = right

    def get_stripes(self):
        """
        Return:
            an array of pairs of y positions, where the first element in the pair
            indicates the start of a non-blank area, and the second element
            indicates the start of the following blank area (might point to the
            first row beyond the bounding box)
        """
        from numpy import array, ediff1d, concatenate, int
        alpha = self.get_channel_data('A')
        max_alpha_per_row = alpha.max(axis=1)

        non_blank_rows = array(max_alpha_per_row != 0, int)
        transitions = ediff1d(concatenate([[0], non_blank_rows, [0]]))
        transition_points = transitions.nonzero()[0].reshape((-1, 2))
        return transition_points

def load_image(infile):
    im = Image.open(infile)
    return ImageProcessor(im)

