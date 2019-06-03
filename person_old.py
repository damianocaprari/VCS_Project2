import numpy as np
import cv2
from parameters import Parameters as P

class PersonOLD(object):

    def __init__(self, coordinates):
        self.id = 0
        self.p1 = np.round(coordinates[:2]).astype(np.int)
        self.p2 = np.round(coordinates[2:]).astype(np.int)
        self.h = self.p2[1] - self.p1[1]
        self.w = self.p2[0] - self.p1[0]
        self.centroid = np.mean([self.p2, self.p1], axis=0)
        self.centroid_past = []
        self.centroid_future = (0,0)
        self.sift_kp= []
        self.sift_descriptors = []    # list of arrays, 1 row for each kp
        # TODO OPTICAL FLOW

    @property
    def color(self):
        return P.COLORS[self.id % (len(P.COLORS))]


    def draw_bounding_box_on_img(self, img):
        img = cv2.rectangle(img, tuple(self.p1), tuple(self.p2), self.color)
        img = cv2.putText(img, 'ID: ' + str(self.id), tuple(self.p1), cv2.FONT_HERSHEY_PLAIN, 0.8, self.color)
        return img

def find_closest_person(current_person, persons):
    current_centroid = current_person.centroid
    others_centroid = np.empty(shape=(len(persons), 2), dtype=np.float)
    for idx, p in enumerate(persons):
        others_centroid[idx] = p.centroid
    # distances = np.sqrt(np.sum(np.square(others_centroid - current_centroid), axis=1)) # no need to calc the sqrt
    distances = np.sum(np.square(others_centroid - current_centroid), axis=1)
    return np.argmin(distances)


def set_sift_keypoints(img, person):
    sift = cv2.xfeatures2d.SIFT_create()
    x11, y11 = person.p1
    x12, y12 = person.p2
    x11 = max(0, x11)
    y11 = max(0, y11)
    x12 = max(0, x12)
    y12 = max(0, y12)
    crop_img = img[y11:y12, x11:x12]
    gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray_crop, None)
    kp, des1 = sift.compute(gray_crop, kp)
    person.sift_kp = kp
    for r in range(des1.shape[0]):
        person.sift_descriptors.append(des1[r, :])
    return person