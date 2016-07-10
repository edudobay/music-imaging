from imaging.core.image import *
from imaging.core.page import *
from imaging.core.signal import average_over_rectangles
import os
from PIL import Image
import numpy
import math

class ImageProcessor:
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

def load_image(infile):
    im = Image.open(infile)
    return ImageProcessor(im)

