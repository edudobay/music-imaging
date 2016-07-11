#!/usr/bin/env python3
import os
from imaging.core.units import Distance
from imaging.core.page import center_on_page
from imaging.core.image import get_channel_data, offset_image
from imaging.core.signal import find_bbox, split_into_nonempty_areas

from PIL import Image # type: ignore

def get_black_level(image):
    return 1 - get_channel_data(image, 'L') / 255

def get_opacity(image):
    return get_channel_data(image, 'A') / 255

def get_stripes(image, *, min_staff_height, min_area_gap, page_crop_width=0):
    image = image.convert('LA')
    black = get_black_level(image)
    opacity = get_opacity(image)

    bbox_left, bbox_right = find_bbox(black, axis=0)

    width, _ = image.size
    if page_crop_width != 0:
        width = int(round(page_crop_width))

    stripes = split_into_nonempty_areas(black, axis=1,
                                        discard_areas_shorter_than = min_staff_height,
                                        minimum_area_gap = min_area_gap)

    bbox_target_x, _ = center_on_page((bbox_right - bbox_left, 0), (width, 0))
    x_offset = bbox_target_x - bbox_left
    new_image = offset_image(image, (x_offset, 0))
    print(bbox_left, bbox_target_x, x_offset, width)

    return [image.crop((-x_offset, top, width - x_offset, bottom)) for (top, bottom) in stripes]

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('infile', metavar='FILE', nargs='+', help='input image file')
    parser.add_argument('-O', metavar='DIR', default=None, dest='output_dir', help='destination directory')
    parser.add_argument('-d', '--dpi', type=float, dest='input_dpi', help='set DPI of input image')
    parser.add_argument('-m', '--min-staff-height', type=Distance, dest='min_staff_height', help='minimum staff height to detect')
    parser.add_argument('-g', '--min-gap', type=Distance, dest='min_area_gap', help='minimum inter-staff gap height to detect')
    parser.add_argument('-W', '--page-width', type=Distance, dest='page_crop_width', help='set output page width')
    args = parser.parse_args()

    def get_logical(value):
        return 0 if not value else value.get_logical(dpi=args.input_dpi)

    for infile in args.infile:
        with Image.open(infile) as im:
            print(infile)
            images = get_stripes(im,
                min_staff_height    = get_logical(args.min_staff_height),
                min_area_gap        = get_logical(args.min_area_gap),
                page_crop_width     = get_logical(args.page_crop_width),
            )

            basename = os.path.splitext(os.path.basename(infile))[0] + '_%02d.png'
            dirname = os.path.dirname(infile) if args.output_dir is None else args.output_dir
            outfile = os.path.join(dirname, basename)
            for i, image in enumerate(images):
                image.save(outfile % i)

