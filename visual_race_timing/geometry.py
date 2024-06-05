import numpy as np


def orientation(p, q, r):
    """Find the orientation of the ordered triplet (p, q, r).
    0 -> p, q and r are collinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    val = (q[:, 1] - p[:, 1]) * (r[:, 0] - q[:, 0]) - (q[:, 0] - p[:, 0]) * (r[:, 1] - q[:, 1])
    return np.where(val == 0, 0, np.where(val > 0, 1, 2))


def on_segment(p, q, r):
    """Check if point q lies on line segment pr."""
    return np.all([
        q[:, 0] <= np.maximum(p[:, 0], r[:, 0]),
        q[:, 0] >= np.minimum(p[:, 0], r[:, 0]),
        q[:, 1] <= np.maximum(p[:, 1], r[:, 1]),
        q[:, 1] >= np.minimum(p[:, 1], r[:, 1])
    ], axis=0)


def do_intersect(p1, q1, p2, q2):
    """Check if line segments 'p1q1' and 'p2q2' intersect."""
    p1, q1, p2, q2 = map(np.array, (p1, q1, p2, q2))

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    intersect = (o1 != o2) & (o3 != o4)

    # Special cases
    collinear = (
            (o1 == 0) & on_segment(p1, p2, q1) |
            (o2 == 0) & on_segment(p1, q2, q1) |
            (o3 == 0) & on_segment(p2, p1, q2) |
            (o4 == 0) & on_segment(p2, q1, q2)
    )

    return intersect | collinear


def line_segment_intersects_boxes(p1, p2, boxes):
    """Check if a line segment intersects with multiple bounding boxes."""
    p1, p2 = np.atleast_2d(p1), np.atleast_2d(p2)
    boxes = np.array(boxes)

    xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Define the four corners of the boxes
    top_left = np.column_stack((xmin, ymin))
    top_right = np.column_stack((xmax, ymin))
    bottom_left = np.column_stack((xmin, ymax))
    bottom_right = np.column_stack((xmax, ymax))

    # Check for intersection with each of the four edges of the boxes
    return (
            do_intersect(p1, p2, top_left, top_right) |
            do_intersect(p1, p2, top_right, bottom_right) |
            do_intersect(p1, p2, bottom_right, bottom_left) |
            do_intersect(p1, p2, bottom_left, top_left)
    )


def side_of_line(p1, p2, points):
    """Check which side of the line p1p2 a point is on."""
    p1, p2 = np.atleast_2d(p1), np.atleast_2d(p2)
    points = np.array(points)

    return np.where(
        (p2[:, 0] - p1[:, 0]) * (points[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (points[:, 0] - p1[:, 0]) > 0,
        1,
        0
    )
