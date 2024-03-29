import math
import numpy
import scipy.signal

DEFAULT_MEDFILT_KERNEL_SIZE = 5
NONEMPTY_REGION_THRESHOLD = 0.03

def average_subsets(ary, split_points):
    """
    Calculate averages of the data in an array over rectangular subsets
    of it. The subsets are determined by a set of split points for each
    dimension.

    `split_points` must be a sequence, containing a sequence of pairs of
    indices for each dimension of `ary`. Each of these pairs of indices defines
    a projection of the regions into which `ary` will be split.
    """
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

def average_over_rectangles(ary, region_shape, *,
                          medfilt_kernel_size = DEFAULT_MEDFILT_KERNEL_SIZE):
    """
    Calculate averages of `ary` over equal rectangular regions.

    If the overall array shape is not an exact multiple of the region shape,
    the last region in each dimension will be smaller than the others.

    Parameters:

        region_shape -- sequence of dimensions of the rectangle

    Fine-tuning:

        medfilt_kernel_size -- see scipy.signal.medfilt
    """
    from numpy import asarray, column_stack, arange
    from scipy.signal import medfilt

    if len(ary.shape) != 2:
        raise NotImplementedError("only 2-D arrays are supported")
    region_shape = asarray(region_shape)
    if len(region_shape) != len(ary.shape):
        raise ValueError("region_shape must have one element per dimension of ary")

    height, width = ary.shape
    sx, sy = region_shape

    grid_y = arange(0, height + sy, sy)
    grid_x = arange(0, width + sx, sx)

    regions = ( column_stack((grid_y[:-1], grid_y[1:])),
                column_stack((grid_x[:-1], grid_x[1:])), )
    averages = average_subsets(ary, regions)
    averages = medfilt(averages, medfilt_kernel_size)
    return averages

def find_bbox(ary, *,
              axis = 0,
              n_regions = 200,
              threshold = NONEMPTY_REGION_THRESHOLD,
              medfilt_kernel_size = DEFAULT_MEDFILT_KERNEL_SIZE):
    """
    Detect an approximate bounding box of the image data in `ary` across the
    dimension specified by `axis`.

    Return a (left, right) coordinate pair.

    Algorithm parameters:

        n_regions -- split the smallest dimension of the image into that number
        of regions, and use square regions with that dimension as side.

        threshold -- minimum average of a region for it to be considered
        non-empty

        medfilt_kernel_size -- see scipy.signal.medfilt
    """
    from math import ceil
    from numpy import array, nonzero

    if len(ary.shape) != 2:
        raise NotImplementedError("only 2-D arrays are supported")
    height, width = ary.shape

    region_shape = ceil(min(width, height) / n_regions)
    averages = average_over_rectangles(ary, (region_shape, region_shape))

    max_along_axis = averages.max(axis=axis)

    left, right = array(nonzero(max_along_axis > threshold)).flatten()[[1, -1]] * region_shape
    right += region_shape # set right boundary to end of region

    return (left, right)

def split_into_nonempty_areas(ary, axis=1, *,
                              discard_areas_shorter_than = 0,
                              minimum_area_gap = 0):
    """
    Find the split points between empty and nonempty areas of `ary`, where an
    empty area is one with only zeroes. Areas are either vertical (0) or
    horizontal (1) according to the specified `axis`.

    Return an array of pairs of y positions, where the first element in the
    pair indicates the start of a non-blank area, and the second element
    indicates the start of the following blank area.

    discard_areas_shorter_than -- if specified, areas shorter than this value
    are discarded from the result set

    minimum_area_gap -- minimum gap between areas for them to be considered as
    two distinct areas (if not distinct, they are merged)
    """
    from numpy import array, ediff1d, concatenate, roll, infty, int
    max_along_axis = ary.max(axis=axis)

    non_blank_rows = array(max_along_axis != 0, int)
    transitions = ediff1d(concatenate([[0], non_blank_rows, [0]]))
    transition_points = transitions.nonzero()[0].reshape((-1, 2))

    area_start = transition_points[:, 0]
    area_end = transition_points[:, 1]

    if discard_areas_shorter_than > 0:
        area_lengths = area_end - area_start

        large_areas = area_lengths >= discard_areas_shorter_than
        small_areas = ~large_areas

        if minimum_area_gap > 0:
            gap_to_next = concatenate([area_start[1:] - area_end[:-1], [infty]])
            gap_to_previous = roll(gap_to_next, 1) # has +infty in first index

            next_is_large = roll(large_areas, -1)
            previous_is_large = roll(large_areas, 1)

            merge_with_next = small_areas & next_is_large & (gap_to_next < minimum_area_gap)
            merge_with_previous = small_areas & previous_is_large & (gap_to_previous < minimum_area_gap)

            #print('transition_points', transition_points)
            #print('area_lengths', area_lengths)
            #print('large_areas', large_areas)
            #print('merge_with_next', merge_with_next)
            #print('merge_with_previous', merge_with_previous)
            #print('gap_to_next', gap_to_next)
            #print('gap_to_previous', gap_to_previous)

            # do the merge. Unused areas will be discarded in the next step
            area_start[roll(merge_with_next, 1)] = area_start[merge_with_next]
            area_end[roll(merge_with_previous, -1)] = area_end[merge_with_previous]

            #print('area_start', area_start)
            #print('area_end', area_end)

        transition_points = transition_points[large_areas]

    return transition_points
