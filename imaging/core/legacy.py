from imaging.core.image import *
from imaging.core.page import *
from imaging.core.signal import average_over_rectangles
import os
from PIL import Image
import numpy
import math

class ImageProcessor:
    def __init__(self, image: Image.Image) -> None:
        self.image = image
        self.width, self.height = image.size # type: int, int
        self.bbox_left, self.bbox_right = 0, self.width
        self.channels = get_image_channels(image)
        self.channel_data = {}

    def get_channel_data(self, *channel_names: str, as_dict: bool = False):
        return get_channel_data(self.image, *channel_names, as_dict=as_dict)

    def extract_stripe(self, top, bottom, *, crop=False):
        bbox_width = self.bbox_right - self.bbox_left
        bbox_target_x, _ = center_on_page((bbox_width, 0), (self.width, 0))
        x_offset = bbox_target_x - self.bbox_left

        board = Image.new('RGBA', (self.width, bottom - top), 'white')
        pasted = board.copy()
        if crop:
            pasted.paste(self.image.crop(( self.bbox_left, top, self.bbox_right, bottom )), (int(round(bbox_target_x)), 0))
        else:
            pasted.paste(self.image.crop(( 0, top, self.width, bottom )), (int(round(x_offset)), 0))

        board = Image.alpha_composite(board, pasted)
        return board.convert('LA')

    def extract_stripes(self, stripes, *, crop=False):
        return [self.extract_stripe(top, bottom, crop=crop) for (top, bottom) in stripes]

    def average_regions(self, region_size):
        blended = alpha_blend_over_background(self.image, 'white').convert('LA')
        im = 1 - get_channel_data(blended, 'L') / 255
        return average_over_rectangles(im, (region_size, region_size))

def load_image(infile):
    im = Image.open(infile)
    return ImageProcessor(im)

