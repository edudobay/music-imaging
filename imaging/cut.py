#!/usr/bin/env python3
import os
from imaging.core import *
from PIL import Image # type: ignore
import numpy as np # type: ignore
from argparse import *

np.set_printoptions(threshold=np.nan)

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

