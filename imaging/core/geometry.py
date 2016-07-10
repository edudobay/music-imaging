def intersect(r1, theta1, r2, theta2):
    from numpy import sin, cos, matrix, array, sqrt
    NORM_TOLERANCE = 1e-7

    s = sin(theta2 - theta1)
    c = cos(theta2 - theta1)
    if s == 0:
        raise ValueError("lines do not intersect")

    rotate90 = matrix([[0, 1], [-1, 0]])

    # unit vectors for both lines
    # r = vector from origin to point on line that is closest to origin
    # t = vector parallel to the line, pointing to increasing theta
    r1_unit = array([cos(theta1), sin(theta1)])
    r2_unit = array([cos(theta2), sin(theta2)])
    t1_unit = r1_unit * rotate90
    t2_unit = r2_unit * rotate90

    # r1 ^r1 + t1 ^t1 = r2 ^r2 + t2 ^t2
    # solving this we get:
    t1, t2 = ((r2, -r1) * matrix([[1, c], [c, 1]]) / s).flat

    p1 = r1 * r1_unit + t1 * t1_unit
    p2 = r2 * r2_unit + t2 * t2_unit
    # p1 and p2 should be the same point
    d = p1 - p2
    assert(sqrt((d * d.T).sum()) < NORM_TOLERANCE)

    return p1

def length(v):
    """Calculate the Euclidean length of a vector"""
    from numpy import sum, sqrt
    return sqrt(sum(v ** 2))

def perspective_transform_coeffs(pa, pb):
    """
    Compute the coefficients of a perspective transformation (as understood by
    PIL) so that a set of four points on plane A get mapped to another set of
    four points on plane B. Could be used as follows:

    >>> from PIL import Image
    >>> coeffs = perspective_transform_coeffs(pa, pb)
    >>> image.transform(image_size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)

    The arguments must be sequences of 4 points with 2 coordinates each,
    specifying the points that describe the mapping.

    Algorithm posted by mmgp at <http://stackoverflow.com/a/14178717/302264>.
    """
    import numpy
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(pb).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)

def polygon_edges(corners):
    """
    Given a sequence of points, return the vector differences between
    consecutive points, including the vector from the last to the first point.
    If the input is the sequence of vertices of a polygon, the result is the
    sequence of the polygon's directed edges.
    """
    from numpy import roll
    sides = roll(corners, -1, axis=0) - corners
    return sides

