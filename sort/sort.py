import numpy as np

from .detections import associate_detections_to_trackers


class SORT(object):

    def __init__(self, tracker_class, max_age=3, min_hits=3):
        self.tracker_class = tracker_class
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        Update the representation of detections
        :param detections: np.array in the format [x1, y1, x2, y2, score]
        :return: The list of indexes that we are certain that are the targets
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_detections, unmatched_trackers = associate_detections_to_trackers(detections, trks)

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trackers:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(detections[d, :][0])

        for i in unmatched_detections:
            trk = self.tracker_class(detections[i])
            self.trackers.append(trk)

        i = len(self.trackers)

        ret = np.zeros(len(self.trackers), dtype=np.long) - 1
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret[i] = i
            if trk.time_since_update > self.max_age:
                ret = np.delete(ret, i)
                ret[-i:] -= 1
                self.trackers.pop(i)

        return ret[ret >= 0]
