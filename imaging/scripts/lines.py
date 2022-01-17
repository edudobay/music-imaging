#!/usr/bin/env python3
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from skimage.transform import (hough_line, hough_line_peaks)

from imaging.core.geometry import *
from imaging.core.image import *
from imaging.core.page import *
from imaging.core.units import *

DEG_THRESHOLD = 7.5

DEBUG = False


class AlgorithmDefaults:
    # In some cases, lower thresholds can help detecting weaker lines.
    # For now, I tune this manually after inspecting the results.
    THRESHOLD_START = 0.95
    THRESHOLD_SHRINK = 0.9


@dataclass
class Context:
    dpi: int
    debug_dpi: int
    vertical_threshold_start: float
    vertical_threshold_shrink: float


def find_corners(horizontal_peaks, vertical_peaks):
    from numpy import array, abs

    _, horiz_theta, horiz_d = horizontal_peaks
    _, vert_theta, vert_d = vertical_peaks

    first_row = abs(horiz_d).argmin()
    last_row = abs(horiz_d).argmax()

    first_col = abs(vert_d).argmin()
    last_col = abs(vert_d).argmax()

    # (top left, top right, bottom right, bottom left)
    corners = (
        intersect(horiz_d[first_row], horiz_theta[first_row], vert_d[first_col], vert_theta[first_col]),
        intersect(horiz_d[first_row], horiz_theta[first_row], vert_d[last_col], vert_theta[last_col]),
        intersect(horiz_d[last_row], horiz_theta[last_row], vert_d[last_col], vert_theta[last_col]),
        intersect(horiz_d[last_row], horiz_theta[last_row], vert_d[first_col], vert_theta[first_col])
    )

    corners = array(corners).reshape((4, 2))
    return corners

def get_target_size(sides):
    from numpy import sqrt, cross
    total_area = (abs(cross(sides[0], sides[3])) + abs(cross(sides[1], sides[2]))) / 2

    mean_width = (length(sides[0]) + length(sides[2])) / 2
    mean_height = (length(sides[1]) + length(sides[3])) / 2

    if DEBUG:
        print('Total area:', total_area, ' --- Mean size:', mean_width, mean_height)
    growth = sqrt(total_area / (mean_width * mean_height))
    target_width = mean_width * growth
    target_height = mean_height * growth
    return target_width, target_height

def find_target_corners(target_pos, target_size):
    from numpy import array, asarray
    target_pos = asarray(target_pos).reshape((1, -1))
    target_width, target_height = target_size
    corners = array([ [0, 0], [target_width, 0],
        [target_width, target_height], [0, target_height] ])
    return target_pos + corners


def find_hough_line_peaks(image, angles, *, min_distance=10, threshold_fraction=0.5):
    from numpy import max
    h, theta, d = hough_line(image, angles)
    peak_h, peak_theta, peak_d = hough_line_peaks(
        h, theta, d,
        min_distance = min_distance, # Minimum distance separating lines
        threshold = threshold_fraction * max(h), # Minimum intensity of peaks
    )
    return peak_h, peak_theta, peak_d


def find_peaks(image, angles, context: Context):
    MIN_DIST_AMPLITUDE = min(*image.shape) * 0.50  # Half of the shortest image dimension

    def good_enough(peaks):
        _, _, dist = peaks
        if len(dist) < 2: return False
        elif (dist.max() - dist.min()) < MIN_DIST_AMPLITUDE: return False
        else: return True

    MAX_TRIES = 7
    tries = 0
    threshold_fraction = context.vertical_threshold_start
    while True:
        if DEBUG:
            print('trying with threshold_fraction =', threshold_fraction)
        peaks = find_hough_line_peaks(image, angles, threshold_fraction = threshold_fraction)
        if good_enough(peaks):
            return peaks
        elif tries > MAX_TRIES:
            raise RuntimeError("maximum iterations exceeded")

        tries += 1
        threshold_fraction *= context.vertical_threshold_shrink


def hough_horizontal_and_vertical(image, *, threshold_deg, filename: Optional[Path] = None, context: Context):
    from numpy import deg2rad, pi, linspace

    # Possible angles in which we look for straight lines.
    # The range [-threshold, threshold] is divided evenly into `num_samples` pieces.
    # The threshold represents the maximum angle for a rotated image to be detectable.
    # These angles can be interpreted as relative to the horizontal axis or to the vertical axis.
    # Both searches are made using the same angles.
    angle_delta = deg2rad(threshold_deg)
    num_samples = 100
    angles = linspace(-angle_delta, angle_delta, num_samples)

    horizontal_peaks = find_hough_line_peaks(image, angles + pi/2)
    vertical_peaks = find_peaks(image, angles, context=context)

    if DEBUG:
        def max_deviation(ary):
            from numpy import abs
            return ary[abs(ary).argmax()]

        from numpy import rad2deg, r_

        print('Report: max angle deviation in horizontal lines:', rad2deg(max_deviation(horizontal_peaks[1] - pi/2)))
        print('Report: max angle deviation in vertical lines:', rad2deg(max_deviation(vertical_peaks[1])))

        print('Horizontal')
        order = horizontal_peaks[0].argsort()
        print(horizontal_peaks[0][order])
        print(horizontal_peaks[1][order])
        print(horizontal_peaks[2][order])

        print('Vertical')
        order = vertical_peaks[0].argsort()
        print(vertical_peaks[0][order])
        print(vertical_peaks[1][order])
        print(vertical_peaks[2][order])

        if DEBUG:
            h, w = image.shape
            plot((w, h),
                 r_[horizontal_peaks[1], vertical_peaks[1]],
                 r_[horizontal_peaks[2], vertical_peaks[2]],
                 filename=filename,
                 dpi=context.debug_dpi,
             )

    return horizontal_peaks, vertical_peaks

def process_image(im, out_filename: Path, *, corners=None, context: Context):
    from numpy import array, mean
    im = im.convert('LA')
    data = get_channel_data(im, 'L')
    data = 1 - data / 255

    source_height, source_width = data.shape
    source_size = source_width, source_height
    page_size = (cm2points(array([21.0, 29.7]), context.dpi) * 1.1).astype(int)

    if corners is None:
        horizontal_peaks, vertical_peaks = hough_horizontal_and_vertical(data, threshold_deg=DEG_THRESHOLD, filename=out_filename, context=context)
        corners = find_corners(horizontal_peaks, vertical_peaks)
    source_centroid = mean(corners, axis=0)
    sides = polygon_edges(corners)

    if DEBUG:
        print('Corners:')
        print(corners)
        print('Sides:')
        print(sides)

    target_size = get_target_size(sides)
    target_centroid_offset = array(target_size) / 2  # from top-left
    target_pos = center_on_page(target_size, page_size)
    target_corners = find_target_corners(target_pos, target_size)
    coeffs = perspective_transform_coeffs(target_corners, corners)

    if DEBUG:
        print('Page size:', page_size)
        print('Source size:', source_size)
        print('Target size:', target_size)
        print('Source centroid:', source_centroid)
        print('Target centroid offset:', target_centroid_offset)
        print('Target position:', target_pos)
        print('Target corners:', target_corners, sep='\n')

    board = Image.new("RGBA", tuple(page_size), (255, 255, 255, 255))

    transformed = im.convert("RGBA").transform(
        tuple(page_size), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    Image.alpha_composite(board, transformed).convert("LA").save(out_filename)

plot_counter = itertools.count(1)

def plot(size, theta, d, *, filename: Optional[Path] = None, dpi: int = 75):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from numpy import cos, sin
    from time import time

    col1, row1 = size
    plt.figure(figsize=(30, 18))
    plt.axes().set_aspect('equal', 'box')

    for angle, dist in zip(theta, d):
        y0 = (dist - 0 * cos(angle)) / sin(angle)
        y1 = (dist - col1 * cos(angle)) / sin(angle)
        plt.plot((0, col1), (y0, y1), '-r')

    plt.xlim(0, col1)
    plt.ylim(0, row1)
    plt.gca().invert_yaxis()


    name = '{}-{:04d}'.format(int(time()), next(plot_counter)) if filename is None \
            else filename.stem
    output = filename.with_stem(f'debug-{name}')
    plt.savefig(output, dpi=dpi)
    plt.close()


def main():
    global DEBUG
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser()
    parser.add_argument('infile', metavar='FILE', nargs='+', help='input image file')
    parser.add_argument('-O', metavar='DIR', default=None, dest='output_dir', help='destination directory')
    parser.add_argument('-d', '--debug', action='store_true', dest='debug')
    parser.add_argument('--dpi', type=int, default=300, help='resolution for input and output images')
    parser.add_argument('--debug-dpi', type=int, default=75, help='resolution for debug plots (default: %(default)s)')
    parser.add_argument('-c', dest='corners', help='a sequence of x,y pairs separated by spaces')
    parser.add_argument('--vertical-threshold-shrink', type=float, default=AlgorithmDefaults.THRESHOLD_SHRINK)
    parser.add_argument('--vertical-threshold-start', type=float, default=AlgorithmDefaults.THRESHOLD_START)
    args = parser.parse_args()

    if args.debug:
        DEBUG = True

    corners = None if not args.corners else \
        [tuple(int(val) for val in point.split(',')) for point in args.corners.split()]

    context = Context(
        dpi=args.dpi,
        debug_dpi=args.debug_dpi,
        vertical_threshold_shrink=args.vertical_threshold_shrink,
        vertical_threshold_start=args.vertical_threshold_start,
    )

    for infile in args.infile:
        with Image.open(infile) as im:
            out_basename = os.path.splitext(os.path.basename(infile))[0] + '.png'
            out_dirname = args.output_dir
            out_filename = Path(out_dirname, out_basename)

            print(infile, '=>', out_filename)
            process_image(im, out_filename, corners=corners, context=context)

if __name__ == '__main__':
    main()
