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

