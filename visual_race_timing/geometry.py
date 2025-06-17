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
    boxes = np.atleast_2d(boxes)

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


def point_to_line_distance(points, p1, p2):
    """Calculate the distance from points to the line segment p1p2 in batch."""
    points, p1, p2 = map(np.array, (points, p1, p2))
    line_vec = p2 - p1
    point_vec = points - p1
    line_len = np.linalg.norm(line_vec, axis=1, keepdims=True)
    line_unitvec = line_vec / line_len
    proj_len = np.einsum('ij,ij->i', point_vec, line_unitvec)
    proj_vec = proj_len[:, np.newaxis] * line_unitvec

    nearest_point = p1 + proj_vec
    nearest_point = np.where(proj_len[:, np.newaxis] < 0, p1, nearest_point)
    nearest_point = np.where(proj_len[:, np.newaxis] > line_len, p2, nearest_point)

    return np.linalg.norm(nearest_point - points, axis=1)


def line_segment_to_box_distance(p1, p2, boxes_xyxy):
    """Calculate the shortest distance from a line segment to multiple bounding boxes in batch."""
    p1, p2 = np.atleast_2d(p1), np.atleast_2d(p2)
    boxes = np.atleast_2d(boxes_xyxy)

    # Check if the line segment intersects any of the boxes
    intersections = line_segment_intersects_boxes(p1, p2, boxes)

    distances = np.full(intersections.shape, np.inf)

    # For each box, calculate the minimum distance to the edges if there's no intersection
    xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    top_left = np.column_stack((xmin, ymin))
    top_right = np.column_stack((xmax, ymin))
    bottom_left = np.column_stack((xmin, ymax))
    bottom_right = np.column_stack((xmax, ymax))

    edges = [
        (top_left, top_right),
        (top_right, bottom_right),
        (bottom_right, bottom_left),
        (bottom_left, top_left)
    ]

    for p1_edge, p2_edge in edges:
        d1 = point_to_line_distance(p1_edge, p1, p2)
        d2 = point_to_line_distance(p2_edge, p1, p2)
        distances = np.minimum(distances, np.minimum(d1, d2))

    return np.where(intersections, 0, distances)
