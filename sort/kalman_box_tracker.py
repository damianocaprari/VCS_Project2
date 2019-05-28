import uuid

import numpy as np
from filterpy.kalman import KalmanFilter

from .utils import convert_bbox_to_sort_representation, convert_sort_representation_to_bbox


class KalmanBoxTracker(object):
    """
    This class is the base class for the tracked objects and contains its state.
    """

    def __init__(self, bbox):
        """
        Initialize the tracker using the initial bbox
        :param bbox: The initial bounding box
        """
        self.kalman_filter = KalmanFilter(dim_x=7, dim_z=4) # Constant velocity model
        self.kalman_filter.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kalman_filter.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kalman_filter.R[2:, 2:] *= 10.
        self.kalman_filter.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kalman_filter.P *= 10.
        self.kalman_filter.Q[-1, -1] *= 0.01
        self.kalman_filter.Q[4:, 4:] *= 0.01

        self.kalman_filter.x[:4] = convert_bbox_to_sort_representation(bbox[:4]).reshape((4, -1))
        self.id = uuid.uuid4()
        self.time_since_update = 0
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the object with the observed bbox
        :param bbox:
        :return:
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kalman_filter.update(convert_bbox_to_sort_representation(bbox[:4]))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate
        :return:
        """
        if (self.kalman_filter.x[6] + self.kalman_filter.x[2]) <= 0:
            self.kalman_filter.x[6] *= 0.0
        self.kalman_filter.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_sort_representation_to_bbox(self.kalman_filter.x).reshape((-1, 4)))
        return self.history[-1]

    @property
    def state(self):
        """
        Returns the current bounding box estimate
        :return:
        """
        return convert_sort_representation_to_bbox(self.kalman_filter.x)
