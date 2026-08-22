
import numpy as np


def has_common_intersection(boxes):
    """
    Checks if N bounding boxes have a common intersection.

    Parameters:
        boxes (np.ndarray): Array of shape (N, 4),
                            where each box is [x_min, y_min, x_max, y_max]

    Returns:
        bool: True if there is a common intersection area, False otherwise.
    """
    if boxes.size == 0:
        return False

    # Compute intersection coordinates
    x_min_inter = np.max(boxes[:, 0])
    y_min_inter = np.max(boxes[:, 1])
    x_max_inter = np.min(boxes[:, 2])
    y_max_inter = np.min(boxes[:, 3])

    # Check for valid intersection
    return (x_max_inter > x_min_inter) and (y_max_inter > y_min_inter)



def get_iou(bbox1, bbox2):
    intersection = np.maximum(np.minimum(bbox1[:, 1], bbox2[:, 1]) - np.maximum(bbox1[:, 0], bbox2[:, 0]), np.zeros_like(bbox1[:, 0]))
    union = np.maximum(bbox1[:, 1], bbox2[:, 1]) - np.minimum(bbox1[:, 0], bbox2[:, 0])
    iou = np.prod(intersection, 1) / np.prod(union, 1)
    return iou
