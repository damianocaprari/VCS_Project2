import numpy as np
import cv2

from sort import KalmanBoxTracker


class Person(KalmanBoxTracker):

    def __init__(self, coordinates):
        super(Person, self).__init__(coordinates)
        self.color = []
        self.__generate_color_from_id()

    def __generate_color_from_id(self):
        id_hex = self.id.hex
        self.color.append(self.__generate_channel_from_hex(id_hex[:8]))
        self.color.append(self.__generate_channel_from_hex(id_hex[8:24]))
        self.color.append(self.__generate_channel_from_hex(id_hex[24:]))

    def __generate_channel_from_hex(cls, hex_string):
        ch = int(hex_string, 16)
        return ch % 256

    def draw_bounding_box_on_img(self, img):
        bbox = self.state
        p1 = tuple(bbox[:2])
        p2 = tuple(bbox[2:])
        img = cv2.rectangle(img, p1, p2, self.color, 2)
        img = cv2.putText(img, 'ID: ' + str(self.id), p1, cv2.FONT_HERSHEY_PLAIN, 0.8, self.color)
        return img


def find_closest_person(current_person, persons):
    current_centroid = current_person.centroid
    others_centroid = np.empty(shape=(len(persons), 2), dtype=np.float)
    for idx, p in enumerate(persons):
        others_centroid[idx] = p.centroid
    # distances = np.sqrt(np.sum(np.square(others_centroid - current_centroid), axis=1)) # no need to calc the sqrt
    distances = np.sum(np.square(others_centroid - current_centroid), axis=1)
    return np.argmin(distances)
