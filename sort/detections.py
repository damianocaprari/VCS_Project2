import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

from .utils import IoU


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Function which associate detected objects to trackers, both represented as bounding boxes
    :param detections:
    :param trackers:
    :param iou_threshold:
    :return:
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=np.int), np.arange(len(detections)), np.empty((0, 5), dtype=np.int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = IoU(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape((1, 2)))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=np.int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
