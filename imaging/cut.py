#!/usr/bin/env python3
import os
from imaging.core import *
from PIL import Image # type: ignore
import numpy as np # type: ignore
import math
from itertools import count
from argparse import *
from typing import *
from scipy.signal import medfilt

np.set_printoptions(threshold=np.nan)

ImageDataPerChannel = Dict[str, np.ndarray]

def get_image_channels(image: Image.Image) -> Dict[str, int]:
    return dict(zip(image.getbands(), count()))

def get_channel_data(image: Image.Image, *channel_names: str, as_dict: bool = False) -> Union[np.ndarray, ImageDataPerChannel]:
    width, height = image.size
    channels = get_image_channels(image)
    channel_data = {}
    for channel_name in channel_names:
        channel_index = channels[channel_name]
        channel_data[channel_name] = np.array(image.getdata(channel_index)).reshape(height, width)
    if as_dict or len(channel_names) != 1:
        return dict(zip(channel_names, (channel_data[channel_name] for channel_name in channel_names)))
    else:
        return channel_data[channel_names[0]] 


def average_regions(ary, split_points):
    ary = np.asarray(ary)
    split_points = [np.asarray(points) for points in split_points]

    if len(split_points) != len(ary.shape):
        raise ValueError("`split_points` must contain a sequence of indices for each dimension of `ary`")
    dim = len(split_points)

    result_shape = tuple(len(points) for points in split_points)
    average = np.empty(shape=result_shape)

    for index in np.ndindex(result_shape):
        region = [slice(*split_points[d][index[d]]) for d in range(dim)]
        average[index] = ary[region].mean()

    return average

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    parser = ArgumentParser()
    parser.add_argument('infile', metavar='FILE', nargs='+', help='input image file')
    parser.add_argument('-O', metavar='DIR', default=None, dest='output_dir', help='destination directory')
    parser.add_argument('-c', '--crop', action='store_true', dest='crop', help='crop images to detected bbox')
    args = parser.parse_args()

    for infile in args.infile:
        with Image.open(infile) as im:
            print(infile)
            ip = ImageProcessor(im)
            ip.find_bbox()
            ip.width = 2480

            images = ip.extract_stripes(ip.get_stripes(), crop=args.crop)
            
            basename = os.path.splitext(os.path.basename(infile))[0] + '_%02d.png'
            dirname = os.path.dirname(infile) if args.output_dir is None else args.output_dir
            outfile = os.path.join(dirname, basename)
            for i, image in enumerate(images):
                image.save(outfile % i)

