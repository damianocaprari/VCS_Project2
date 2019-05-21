import numpy as np
import cv2


class Person(object):

    def __init__(self, coordinates, color):
        self.id = id(self)
        self.p1 = np.round(coordinates[:2]).astype(np.int)
        self.p2 = np.round(coordinates[2:]).astype(np.int)
        self.h = self.p2[1] - self.p1[1]
        self.w = self.p2[0] - self.p1[0]
        self.centroid = np.mean([self.p2, self.p1], axis=0)
        self.color = color

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
