import numpy as np


def IoU(bbox_test, bbox_gt):
    """
    Computes the IOU between two bbox in the form [x1, y1, x2, y2]
    :param bbox_test: The test bounding box
    :param bbox_gt: The Ground Truth bounding box
    :return: the IOU between the two boxes
    """
    xx1 = np.maximum(bbox_test[0], bbox_gt[0])
    yy1 = np.maximum(bbox_test[1], bbox_gt[1])
    xx2 = np.minimum(bbox_test[2], bbox_gt[2])
    yy2 = np.minimum(bbox_test[3], bbox_gt[3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w*h

    iou = wh / ((bbox_test[2]-bbox_test[0])*(bbox_test[3]-bbox_test[1]) + (bbox_gt[2]-bbox_gt[0])*(bbox_gt[3]-bbox_gt[1]) - wh)
    return iou


def convert_bbox_to_sort_representation(bbox):
    """
    Convert a bounding box in the form [x1, y1, x2, y2] to the sort representation [x, y, s, r]
    - x: horizontal coordinate of the center of the bounding box
    - y: vertical coordinate of the center of the bounding box
    - s: scale/area of the bounding box
    - r: ratio of the bounding box
    :param bbox: The bounding box
    :return: Sort representation of bounding box
    """
    bbox = bbox.reshape((2, 2))
    w, h = np.diff(bbox, axis=0).reshape(-1)
    x = bbox[0, 0] + w / 2
    y = bbox[0, 1] + h / 2
    s = w*h
    r = w / np.float(h)
    return np.array([x, y, s, r], dtype=np.float)


def convert_sort_representation_to_bbox(sort_repr):
    """
    Converts a sort representation of the bounding box [x, y, s, r] to a representation with [x1, y1, x2, y2].
    :param sort_repr: sort representation of the bounding box
    :return: A geometric representation of bounding box
    """
    w = np.sqrt(sort_repr[2] * sort_repr[3])
    h = sort_repr[2] / w
    x1 = sort_repr[0] - w / 2
    y1 = sort_repr[1] - h / 2
    x2 = sort_repr[0] + w / 2
    y2 = sort_repr[1] + h / 2
    return np.array([x1, y1, x2, y2], dtype=np.float)


