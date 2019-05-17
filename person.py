import numpy as np
import cv2


class Person(object):

    __NEXT_ID = 0

    def __init__(self, coordinates, color):
        self.id = self.__NEXT_ID
        self.__NEXT_ID += 1
        self.p1 = np.round(coordinates[:2]).astype(np.int)
        self.p2 = np.round(coordinates[2:]).astype(np.int)
        self.h = self.p2[1] - self.p1[1]
        self.w = self.p2[0] - self.p1[0]
        mean_displacement = np.mean(self.p2 - self.p1, axis=0)
        self.centroid = self.p1 + mean_displacement
        self.color = color

    def draw_bounding_box_on_img(self, img):
        img = cv2.rectangle(img, tuple(self.p1), tuple(self.p2), self.color)
        return img
