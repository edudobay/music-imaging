#!/usr/bin/env python3
from imaging.core.signal import *
from imaging.core.units import *
from imaging.core.geometry import *
from imaging.core.page import *
from imaging.core.image import *

from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
import numpy

DEG_THRESHOLD = 7.5

DEBUG = False

def find_corners(horizontal_peaks, vertical_peaks):
    from numpy import array, rad2deg, abs

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
        min_distance = min_distance,
        threshold = threshold_fraction * max(h))
    return peak_h, peak_theta, peak_d

def hough_horizontal_and_vertical(image, *, threshold_deg):
    from numpy import deg2rad, pi, linspace
    angle_delta = deg2rad(threshold_deg)
    angles = linspace(-angle_delta, angle_delta, 100)

    def find_peaks(image, angles):
        MIN_DIST_AMPLITUDE = min(*image.shape) * 0.50
        def good_enough(peaks):
            _, _, dist = peaks
            if len(dist) < 2: return False
            elif (dist.max() - dist.min()) < MIN_DIST_AMPLITUDE: return False
            else: return True

        threshold_fraction = 0.95
        MAX_TRIES = 7
        tries = 0
        while True:
            if DEBUG:
                print('trying with threshold_fraction =', threshold_fraction)
            peaks = find_hough_line_peaks(image, angles, threshold_fraction = threshold_fraction)
            if good_enough(peaks):
                return peaks
            elif tries > MAX_TRIES:
                raise RuntimeError("maximum iterations exceeded")
                
            tries += 1
            threshold_fraction *= 0.9

    horizontal_peaks = find_hough_line_peaks(image, angles + pi/2)
    vertical_peaks = find_peaks(image, angles)

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
                 r_[horizontal_peaks[2], vertical_peaks[2]])

    return horizontal_peaks, vertical_peaks

def process_image(im, out_filename):
    from numpy import array, linspace, max, mean, pi
    im = im.convert('LA')
    data = get_channel_data(im, 'L')
    data = 1 - data / 255
    DPI = 300

    source_height, source_width = data.shape
    source_size = source_width, source_height
    page_size = (cm2points(array([21.0, 29.7]), DPI) * 1.1).astype(int)

    horizontal_peaks, vertical_peaks = hough_horizontal_and_vertical(data, threshold_deg=DEG_THRESHOLD)
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
    target_pos = source_centroid - target_centroid_offset + center_on_page(source_size, page_size)
    coeffs = perspective_transform_coeffs(find_target_corners(target_pos, target_size), corners)

    board = Image.new("RGBA", tuple(page_size), (255, 255, 255, 255))

    transformed = im.convert("RGBA").transform(
        tuple(page_size), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    Image.alpha_composite(board, transformed).convert("LA").save(out_filename)

def plot(size, theta, d):
    import matplotlib.pyplot as plt
    from numpy import cos, sin
    col1, row1 = size
    plt.figure(figsize=(30, 18))

    for angle, dist in zip(theta, d):
        y0 = (dist - 0 * cos(angle)) / sin(angle)
        y1 = (dist - col1 * cos(angle)) / sin(angle)
        plt.plot((0, col1), (y0, y1), '-r')

    plt.xlim(0, col1)
    plt.ylim(0, row1)
    plt.gca().invert_yaxis()
    plt.axes().set_aspect('equal', 'box')

    plt.savefig('debug.png', dpi=200)

if __name__ == '__main__':
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser()
    parser.add_argument('infile', metavar='FILE', nargs='+', help='input image file')
    parser.add_argument('-O', metavar='DIR', default=None, dest='output_dir', help='destination directory')
    parser.add_argument('-d', action='store_true', dest='debug')
    args = parser.parse_args()

    if args.debug:
        DEBUG = True

    for infile in args.infile:
        with Image.open(infile) as im:
            out_basename = os.path.splitext(os.path.basename(infile))[0] + '.png'
            out_dirname = args.output_dir
            out_filename = os.path.join(out_dirname, out_basename)

            print(infile, '=>', out_filename)
            process_image(im, out_filename)

