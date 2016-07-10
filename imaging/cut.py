#!/usr/bin/env python3
import os
from imaging.core.page import center_on_page
from imaging.core.image import get_channel_data, offset_image
from imaging.core.signal import find_bbox, split_into_nonempty_areas

from PIL import Image # type: ignore
import numpy as np # type: ignore
from argparse import *

np.set_printoptions(threshold=np.nan)

def get_black_level(image):
    return 1 - get_channel_data(image, 'L') / 255

def get_opacity(image):
    return get_channel_data(image, 'A') / 255

def get_stripe(image, top, bottom):
    return image.crop(0, top, image.width, bottom)

def get_stripes(image):
    image = image.convert('LA')
    black = get_black_level(image)
    opacity = get_opacity(image)

    bbox_left, bbox_right = find_bbox(black, axis=0)

    stripes = split_into_nonempty_areas(opacity, axis=1)

    bbox_target_x, _ = center_on_page((bbox_right - bbox_left, 0), image.size)
    new_image = offset_image(image, (bbox_target_x - bbox_left, 0))

    return [get_stripe(new_image, top, bottom) for (top, bottom) in stripes]

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser.add_argument('infile', metavar='FILE', nargs='+', help='input image file')
    parser.add_argument('-O', metavar='DIR', default=None, dest='output_dir', help='destination directory')
    parser.add_argument('-c', '--crop', action='store_true', dest='crop', help='crop images to detected bbox')
    parser.add_argument('-w', '--width', dest='page_width', help='set output page width (in pixels)')
    args = parser.parse_args()

    for infile in args.infile:
        with Image.open(infile) as im:
            print(infile)
            images = get_stripes(im)
            
            basename = os.path.splitext(os.path.basename(infile))[0] + '_%02d.png'
            dirname = os.path.dirname(infile) if args.output_dir is None else args.output_dir
            outfile = os.path.join(dirname, basename)
            for i, image in enumerate(images):
                image.save(outfile % i)

